import copy, math, importlib
import torch
import torch.nn.functional as nnf
from torch import nn, Tensor
from typing import Dict

from collections import OrderedDict
from transformers import GPT2Config, GPT2LMHeadModel

from pointcept.models.threedreftr.captioner_dcc.helper import Matcher
from pointcept.models.threedreftr.captioner_dcc.generation_utils import generation
from pointcept.models.threedreftr.captioner_dcc.scst import SCST_Training
from pointcept.datasets.scanrefer_jointdc import SCANREFER, ScanReferTokenizer
from pointcept.models.losses.vqa_losses import generalized_box_iou3d, box_cxcyczwhd_to_xyzxyz


def box_cxcyczwhd_to_xyzxyz(x):
    x_c, y_c, z_c, w, h, d = x.unbind(-1)
    w = torch.clamp(w, min=1e-6)
    h = torch.clamp(h, min=1e-6)
    d = torch.clamp(d, min=1e-6)
    assert (w < 0).sum() == 0
    assert (h < 0).sum() == 0
    assert (d < 0).sum() == 0
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (z_c - 0.5 * d),
         (x_c + 0.5 * w), (y_c + 0.5 * h), (z_c + 0.5 * d)]
    return torch.stack(b, dim=-1)


def normalized_center(x):

    min_val_0, _ = torch.min(x, dim=0, keepdim=True)
    min_val, _ = torch.min(min_val_0, dim=1, keepdim=True)
    max_val_0, _ = torch.max(x, dim=0, keepdim=True)
    max_val, _ = torch.max(max_val_0, dim=1, keepdim=True)

    return (x - min_val) / (max_val - min_val)


@torch.no_grad()
def hungarian_matching(matcher: Matcher, end_points: dict, targets: dict) -> dict:
    
    outputs = end_points.copy()
    nactual_gt = targets["box_label_mask"].sum(axis=1).long()
    num_boxes = torch.clamp(nactual_gt.sum(), min=1).item()
    
    targets["nactual_gt"] = nactual_gt
    targets["num_boxes"] = num_boxes
    targets["num_boxes_replica"] = nactual_gt.sum().item()

    out_bbox = torch.cat([outputs['last_center'], outputs['last_pred_size']], dim=-1)  # [8, 256, 6]
    tgt_bbox = torch.cat([targets['center_label'], targets['size_gts']], dim=-1)  # [8, 132, 6]

    # for match only here
    outputs["gious"] = torch.stack([
            generalized_box_iou3d(
                box_cxcyczwhd_to_xyzxyz(o),
                box_cxcyczwhd_to_xyzxyz(t)
            ) for o, t in zip(out_bbox, tgt_bbox)
        ], dim=0)  # [8, 256, 132]

    center_dist = torch.cdist(
        normalized_center(outputs["last_center"]), 
        normalized_center(targets["center_label"]), p=1
    )  # [8, 256, 132]
    outputs["center_dist"] = center_dist
    
    return matcher(outputs, targets)


def proposal_dimension_select(features: Tensor, indices: Tensor) -> Tensor:
    '''
    
    Parameters
    ----------
    features : Tensor, with size [batch x nsrc x ...]
        Data bank, from which to gather information.
    indices : Tensor, with size [batch x ntgt]
        Indices for gathering information from data bank.

    Returns
    -------
    Tensor, with size [batch x ntgt x ...]
        Gathers features in proposal dimension.
    
    '''
    return torch.gather(
        features, 1, 
        indices.reshape(
            *(indices.shape + tuple(1 for _ in features.shape[2:]))
        ).repeat(
            *((1, 1) + features.shape[2:])
        )
    )


def decode_box_corners(box_corners: Tensor) -> Tensor:
    box_corners = copy.deepcopy(box_corners.detach())
    box_corners[..., [1, 2]] = box_corners[..., [2, 1]]
    box_corners[..., -1] *= -1
    return box_corners


def position_embedding(max_len: int, d_model: int) -> Tensor:
    position_embedding = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                         -(math.log(10000.0) / d_model))
    position_embedding[:, 0::2] = torch.sin(position * div_term)
    position_embedding[:, 1::2] = torch.cos(position * div_term)
    return position_embedding


class Captioner(nn.Module):

    def __init__(self):
        super(Captioner, self).__init__()
        
        self.embedding_size = 256
        self.max_positions = 64
        self.max_des_len = 32
        self.use_beam_search = True
        self.use_scst = False  # debug
        
        ## initialize tokenizer for batch decoding
        self.tokenizer = ScanReferTokenizer(SCANREFER['vocabulary']['word2idx'])
        self.nvocabs = len(self.tokenizer)  # 3433
        
        ## for label assignment
        self.matcher = Matcher(
            cost_class=1, cost_objectness=0, cost_giou=2, cost_center=0
        )
        
        ## caption generation cores
        gpt2_config = GPT2Config(
            vocab_size=self.nvocabs,
            n_positions=self.max_positions,
            n_embd=self.embedding_size,
            n_layer=2,
            n_head=4,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            add_cross_attention=True,
        )
        self.transformer = GPT2LMHeadModel(config=gpt2_config)
        self.transformer.transformer.wpe = nn.Embedding.from_pretrained(
            position_embedding(self.max_positions, self.embedding_size)
        )
        
        ## for proposal feature projection
        self.feature_projector = nn.Sequential(
            nn.Linear(288, self.embedding_size),
            nn.LayerNorm(self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size),
        )
        
        self.context_projector = nn.Sequential(
            nn.Linear(288, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(),
        )
        
        ## ---- super parameters for evaluation
        self.caption_config = {
            'early_stopping': True,
            'eos_token_id': self.tokenizer.eos_token_id,
            'num_beams': 5 if self.use_beam_search is True else None,
        }
        
        # self.scst = SCST_Training(args)  debug
        self.use_scst = hasattr(self, 'use_scst') and self.use_scst is True
        self.scst_max_des_per_iter = 32
    
    
    def prepare_object_representations(self, end_points: dict) -> dict:
        
        ## extract proposal feature: batch x nprop x channel
        last_layer_output = end_points['query_last']  # [8, 256, 288]
        object_feature = self.feature_projector(last_layer_output)  # [8, 256, 256]
        prefix_feature = object_feature.unsqueeze(2)  # [8, 256, 1, 256]
        # batch x nprop x 1 x channel, as RNN-like guidance
        end_points['object_features'] = prefix_feature
        
        ## batch x nprop x ntgt x channel, as cross attention guidance
        query_xyz = end_points['query_points_xyz']  # [8, 256, 3]
        batch, nprop, _ = query_xyz.shape
        
        # batch x nprop x npoints
        center_distance = torch.cdist(query_xyz, end_points['fp2_xyz'])  # [8, 256, 1024]
        
        # batch x nprop x k
        k_near_indice = center_distance.topk(k=128, largest=False, dim=-1).indices  # [8, 256, 128]
        k_near_context_feature = proposal_dimension_select(  
            self.context_projector(end_points['fp2_features'].transpose(-2, -1)), 
            k_near_indice.reshape(batch, -1)
        )   # batch x (nprop x k) x channel  [8, 32768, 256]
        k_near_context_feature = k_near_context_feature.reshape(
            batch, nprop, -1, self.embedding_size
        )   # batch x nprop x k x channel  [8, 256, 128, 256]
        
        end_points['k_near_context'] = k_near_indice
        end_points['encoder_hidden_states'] = k_near_context_feature
        
        return end_points
        
    
    def forward(self, end_points: dict, data_dict: dict, is_eval: bool=False) -> dict:
             
        # nlayers x batch x nprop x channel -> batch x nprop x 1 x channel
        end_points = self.prepare_object_representations(end_points)
        
        if is_eval ==  False:
            if self.use_scst is True:
                return self.forward_scst(end_points, data_dict)
            return self.forward_training(end_points, data_dict)
        else:
            return self.forward_evaluation(end_points, data_dict)
        

    
    def forward_training(self, end_points: Dict, data_dict: Dict) -> Dict:
        
        # get word embeddings, NOTE: captioner does not predict <bos> token
        caption_ids = end_points['reference_tokens']    # batch x MAX_NUM_OBJ x ntokens  [8, 132, 32]
        embedding_mask = end_points['reference_masks']  # batch x MAX_NUM_OBJ x ntokens  [8, 132, 32]
        
        # ---- match proposal bounding boxes with ground truth inds
        assignments = hungarian_matching(
            self.matcher, end_points, data_dict
        )
        
        # ---- generate caption labels for rnn model
        gt_box_cap_label = proposal_dimension_select(
            caption_ids, assignments['per_prop_gt_inds'].long()
        )   # batch x nproposals x max_des_len
        gt_box_cap_masks = proposal_dimension_select(
            embedding_mask, assignments['per_prop_gt_inds'].long()
        )   # batch x nproposals x max_des_len
        
        # no loss for objects with background and non annotated objects
        unvalid_proposal = assignments['proposal_matched_mask']  # lack of match proposal?
        unannotated_proposal = (gt_box_cap_label[..., 0] != 0).long()
        annotated_proposal = unvalid_proposal * unannotated_proposal
        assignments['annotated_proposal'] = annotated_proposal
        
        # ---- generate caption embeddings for rnn model
        prefix_tokens = end_points['object_features']
        inputs_embeds = torch.cat([
            prefix_tokens, self.transformer.transformer.wte(gt_box_cap_label)
        ], dim=2)   # batch x nproposals x (nprefix + max_des_len) x channel
        inputs_masks = torch.cat([
            torch.ones_like(prefix_tokens[..., 0]), gt_box_cap_masks
        ], dim=2)   # batch x nproposals x (nprefix + max_des_len)W

        inputs_embeds = inputs_embeds[annotated_proposal == 1]
        inputs_masks = inputs_masks[annotated_proposal == 1]

        outputs = self.transformer( # num_annotated x (1 + max_des_len)
            inputs_embeds=inputs_embeds,
            attention_mask=inputs_masks,
            encoder_hidden_states=\
                None if end_points.get('encoder_hidden_states', None) is None else \
                    end_points['encoder_hidden_states'][annotated_proposal == 1]
        )
        
        return outputs, prefix_tokens, annotated_proposal, gt_box_cap_label
    
    
    def forward_scst(self, detector_output: Dict, inputs: Dict) -> Dict:
        
        # get word embeddings, NOTE: captioner does not predict <bos> token
        caption_ids = inputs['reference_tokens']    # batch x MAX_NUM_OBJ x ntokens
        
        # ---- match proposal bounding boxes with ground truth inds
        assignments = hungarian_matching(
            self.matcher, detector_output, inputs
        )
        
        # ---- generate caption labels for rnn model
        gt_box_cap_label = proposal_dimension_select(
            caption_ids, assignments['per_prop_gt_inds'].long()
        )   # batch x nproposals x max_des_len
        
        # no loss for objects with background and non annotated objects
        unvalid_proposal = assignments['proposal_matched_mask']
        unannotated_proposal = (gt_box_cap_label[..., 0] != 0).long()
        annotated_proposal = unvalid_proposal * unannotated_proposal
        
        if torch.sum(annotated_proposal == 1).cpu().tolist() > self.scst_max_des_per_iter:
            random_value = torch.randn(annotated_proposal.shape, device=annotated_proposal.device)
            random_value[annotated_proposal == 0] = 1e8
            
            random_threshold = torch.kthvalue(
                random_value.view(-1), 
                self.scst_max_des_per_iter
            ).values
            
            annotated_proposal *= (random_value <= random_threshold).long()
        
        assignments['annotated_proposal'] = annotated_proposal
        
        # generation with greedy search
        prefix_tokens = detector_output['object_features']
        
        greedy_caption = generation(
            self.transformer, 
            inputs_embeds=prefix_tokens[annotated_proposal == 1],
            encoder_hidden_states=\
                None if detector_output.get('encoder_hidden_states', None) is None else \
                    detector_output['encoder_hidden_states'][annotated_proposal == 1],
            early_stopping = True,
            eos_token_id = self.tokenizer.eos_token_id,
            num_beams = None,
        )
        
        beam_caption = generation(
            self.transformer, 
            inputs_embeds=prefix_tokens[annotated_proposal == 1],
            encoder_hidden_states=\
                None if detector_output.get('encoder_hidden_states', None) is None else \
                    detector_output['encoder_hidden_states'][annotated_proposal == 1],
            **self.caption_config
        )
        scst_loss = self.scst(greedy_caption, beam_caption, inputs, assignments)
        detector_output['loss'] += 5 * scst_loss
        
        return detector_output
    
    
    def forward_evaluation(self, detector_output: Dict, inputs: Dict) -> Dict:
        
        # proposal_tokens: batch x nprop x nprefix x channel
        prefix_tokens = detector_output['object_features']
        
        batch, nproposals, nprefix, channel = prefix_tokens.shape # torch.Size([8, 256, 1, 256])
        
        caption_output = OrderedDict()
        
        # import pdb; pdb.set_trace()
        for batch_id in range(batch):
            scene_cap_output = generation(
                self.transformer, 
                inputs_embeds=prefix_tokens[batch_id],
                encoder_hidden_states=\
                    None if detector_output.get('encoder_hidden_states', None) is None else \
                        detector_output['encoder_hidden_states'][batch_id],
                **self.caption_config
            )
            # update scene output to batch output
            for key, tensor in scene_cap_output.items():
                caption_output[key] = caption_output.get(key, []) + [tensor]
        
        for key, tensor in caption_output.items():
            caption_output[key] = torch.cat(caption_output[key], dim=0)
        

        captions = self.tokenizer.batch_decode( 
            caption_output['output_ids'].tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        lang_cap = [
            [
                'sos ' + captions[batch_id * nproposals + prop_id] + ' eos' \
                    for prop_id in range(nproposals)
            ] \
            for batch_id in range(batch)
        ]
        
        return lang_cap
