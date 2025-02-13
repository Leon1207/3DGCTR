# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
# Parts adapted from Group-Free
# Copyright (c) 2021 Ze Liu. All Rights Reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizerFast

from .backbone_module import Pointnet2Backbone
from .backbone_ptv2maxpool import PMBEMBAttn
from .backbone_swin3d import Swin3DUNet
from .backbone_spunet import SpUNetBase
from .modules import (
    PointsObjClsModule, GeneralSamplingModule,
    ClsAgnosticPredictHead, PositionEmbeddingLearned
)
from .encoder_decoder_layers import (
    BiEncoder, BiEncoderLayer, BiDecoderLayer
)
from torch_scatter import scatter_mean
from .pointnet2 import pointnet2_utils
from pointcept.models.builder import MODELS


@MODELS.register_module("3dreftr_eda")
class ThreeDRefTR_SP(nn.Module):
    """
    3D language grounder.

    Args:
        num_class (int): number of semantics classes to predict
        num_obj_class (int): number of object classes
        input_feature_dim (int): feat_dim of pointcloud (without xyz)
        num_queries (int): Number of queries generated
        num_decoder_layers (int): number of decoder layers
        self_position_embedding (str or None): how to compute pos embeddings
        contrastive_align_loss (bool): contrast queries and token features
        d_model (int): dimension of features
        butd (bool): use detected box stream
        pointnet_ckpt (str or None): path to pre-trained pp++ checkpoint
        self_attend (bool): add self-attention in encoder
    """

    def __init__(self, num_class=256, num_obj_class=485,
                 input_feature_dim=3,
                 num_queries=256,
                 num_decoder_layers=6, self_position_embedding='loc_learned',
                 contrastive_align_loss=True,
                 d_model=288, butd=True, pointnet_ckpt=None,  # butd
                 data_path="/data/pointcloud/data_for_eda/scannet_others_processed/",
                 self_attend=True):
        """Initialize layers."""
        super().__init__()

        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.self_position_embedding = self_position_embedding
        self.contrastive_align_loss = contrastive_align_loss
        self.butd = butd  # debug

        # Visual encoder
        self.backbone_net = Pointnet2Backbone(input_feature_dim=input_feature_dim, width=1)
        # self.backbone_net = PMBEMBAttn(in_channels=input_feature_dim)
        # self.backbone_net = SpUNetBase(in_channels=input_feature_dim)
        # self.backbone_net = Swin3DUNet(in_channels=input_feature_dim)

        if input_feature_dim == 3 and pointnet_ckpt is not None:
            self.backbone_net.load_state_dict(torch.load(
                pointnet_ckpt
            ), strict=False)

        # Text Encoder
        # # (1) online
        # t_type = "roberta-base"
        # NOTE (2) offline: load from the local folder.
        # t_type = "roberta-base"
        t_type = f'{data_path}/roberta-base/'
        self.tokenizer = RobertaTokenizerFast.from_pretrained(t_type, local_files_only=True)
        self.text_encoder = RobertaModel.from_pretrained(t_type, local_files_only=True, use_safetensors=False)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, d_model),
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Dropout(0.1)
        )

        # Box encoder
        if self.butd:
            self.butd_class_embeddings = nn.Embedding(num_obj_class, 768)
            saved_embeddings = torch.from_numpy(np.load(
                '/userhome/lyd/3dvlm/data/class_embeddings3d.npy', allow_pickle=True
            ))
            self.butd_class_embeddings.weight.data.copy_(saved_embeddings)
            self.butd_class_embeddings.requires_grad = False
            self.class_embeddings = nn.Linear(768, d_model - 128)
            self.box_embeddings = PositionEmbeddingLearned(6, 128)

        # Cross-encoder
        self.pos_embed = PositionEmbeddingLearned(3, d_model)
        bi_layer = BiEncoderLayer(
            d_model, dropout=0.1, activation="relu",
            n_heads=8, dim_feedforward=256,
            self_attend_lang=self_attend, self_attend_vis=self_attend,
            use_butd_enc_attn=butd
        )
        self.cross_encoder = BiEncoder(bi_layer, 3)

        # Mask Feats Generation layer
        self.x_mask = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, 1), 
            nn.ReLU(), 
            nn.Conv1d(d_model * 2, d_model * 2, 1),
            nn.ReLU(), 
            nn.Conv1d(d_model * 2, d_model, 1)
            )
        self.x_query = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, 1), 
            nn.ReLU(), 
            nn.Conv1d(d_model * 2, d_model * 2, 1),
            nn.ReLU(), 
            nn.Conv1d(d_model * 2, d_model, 1)
            )
        self.super_grouper = pointnet2_utils.QueryAndGroup(radius=0.2, nsample=2, use_xyz=False, normalize_xyz=True)

        # Query initialization
        self.points_obj_cls = PointsObjClsModule(d_model)
        self.gsample_module = GeneralSamplingModule()
        self.decoder_query_proj = nn.Conv1d(d_model, d_model, kernel_size=1)

        # Proposal (layer for size and center)
        self.proposal_head = ClsAgnosticPredictHead(
            num_class, 1, num_queries, d_model,
            objectness=False, heading=False,
            compute_sem_scores=True
        )

        # Transformer decoder layers
        self.decoder = nn.ModuleList()
        for _ in range(self.num_decoder_layers):
            self.decoder.append(BiDecoderLayer(
                d_model, n_heads=8, dim_feedforward=256,
                dropout=0.1, activation="relu",
                self_position_embedding=self_position_embedding, butd=self.butd
            ))

        # Prediction heads
        self.prediction_heads = nn.ModuleList()
        for _ in range(self.num_decoder_layers):
            self.prediction_heads.append(ClsAgnosticPredictHead(
                num_class, 1, num_queries, d_model,
                objectness=False, heading=False,
                compute_sem_scores=True
            ))

        # Extra layers for contrastive losses
        if contrastive_align_loss:
            self.contrastive_align_projection_image = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 64)
            )
            self.contrastive_align_projection_text = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 64)
            )

        # Init
        self.init_bn_momentum()
    
    
    # BRIEF visual and text backbones.
    def _run_backbones(self, data_dict):
        """Run visual and text backbones."""
        # step 1. Visual encoder
        end_points = self.backbone_net(data_dict['point_clouds'], data_dict['offset'], data_dict['superpoint'].shape[0])
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = end_points['fp2_xyz']
        end_points['seed_features'] = end_points['fp2_features']
        
        # step 2. Text encoder
        tokenized = self.tokenizer.batch_encode_plus(
            data_dict['text'], padding="longest", return_tensors="pt"
        ).to(data_dict['point_clouds'].device)
        
        encoded_text = self.text_encoder(**tokenized)
        text_feats = self.text_projector(encoded_text.last_hidden_state)

        # Invert attention mask that we get from huggingface
        # because its the opposite in pytorch transformer
        text_attention_mask = tokenized.attention_mask.ne(1).bool()

        end_points['text_feats'] = text_feats
        end_points['text_attention_mask'] = text_attention_mask
        end_points['tokenized'] = tokenized
        return end_points

    # BRIEF generate query.
    def _generate_queries(self, xyz, features, end_points):
        # kps sampling
        points_obj_cls_logits = self.points_obj_cls(features)  # [B, 1, K=1024]
        end_points['seeds_obj_cls_logits'] = points_obj_cls_logits
        
        # top-k
        sample_inds = torch.topk(   
            torch.sigmoid(points_obj_cls_logits).squeeze(1),
            self.num_queries
        )[1].int()

        xyz, features, sample_inds = self.gsample_module(   
            xyz, features, sample_inds
        )

        end_points['query_points_xyz'] = xyz  # (B, V, 3)
        end_points['query_points_feature'] = features  # (B, F, V)
        end_points['query_points_sample_inds'] = sample_inds  # (B, V)
        return end_points
    
    # segmentation prediction
    def _seg_seeds_prediction(self, query, mask_feats, end_points, prefix=''):
        ## generate seed points masks
        pred_mask_seeds = torch.einsum('bnd,bdm->bnm', query, mask_feats)
        ## mapping seed points masks to superpoints masks
        return pred_mask_seeds

    # BRIEF forward.
    def forward(self, data_dict):
        """
        Forward pass.
        Args:
            inputs: dict
                {point_clouds, text}
                point_clouds (tensor): (B, Npoint, 3 + input_channels)
                text (list): ['text0', 'text1', ...], len(text) = B

                more keys if butd is enabled:
                    det_bbox_label_mask
                    det_boxes
                    det_class_ids
        Returns:
            end_points: dict
        """
        # STEP 1. vision and text encoding
        end_points = self._run_backbones(data_dict)
        points_xyz = end_points['fp2_xyz']
        points_features = end_points['fp2_features']
        text_feats = end_points['text_feats']
        text_padding_mask = end_points['text_attention_mask']
        
        # STEP 2. Box encoding
        if self.butd:
            # attend on those features
            detected_mask = ~data_dict['det_bbox_label_mask']

            # step box position.    det_boxes ([B, 132, 6]) -->  ([B, 128, 132])
            box_embeddings = self.box_embeddings(data_dict['det_boxes'])
            # step box class        det_class_ids ([B, 132])  -->  ([B, 132, 160])
            class_embeddings = self.class_embeddings(self.butd_class_embeddings(data_dict['det_class_ids']))
            # step box feature     ([B, 132, 288])
            detected_feats = torch.cat([box_embeddings, class_embeddings.transpose(1, 2)]
                                        , 1).transpose(1, 2).contiguous()
        else:
            detected_mask = None
            detected_feats = None

        # STEP 3. Cross-modality encoding
        points_features, text_feats = self.cross_encoder(
            vis_feats=points_features.transpose(1, 2).contiguous(),
            pos_feats=self.pos_embed(points_xyz).transpose(1, 2).contiguous(),
            padding_mask=torch.zeros(
                len(points_xyz), points_xyz.size(1)
            ).to(points_xyz.device).bool(),
            text_feats=text_feats,
            text_padding_mask=text_padding_mask,
            end_points=end_points,
            detected_feats=detected_feats,
            detected_mask=detected_mask
        )
        points_features = points_features.transpose(1, 2)
        points_features = points_features.contiguous()
        end_points["text_memory"] = text_feats
        end_points['seed_features'] = points_features
        
        # STEP 4. text projection --> 64
        if self.contrastive_align_loss:
            proj_tokens = F.normalize(
                self.contrastive_align_projection_text(text_feats), p=2, dim=-1
            )
            end_points['proj_tokens'] = proj_tokens     # ([B, L, 64])

        # STEP 4.1 Mask Feats Generation
        mask_feats = self.x_mask(points_features).float()  # [B, 288, 1024]
        superpoint = data_dict['superpoint']  # [B, 50000]
        end_points['superpoints'] = superpoint
        source_xzy = data_dict['source_xzy'].contiguous()  # [B, 50000, 3]
        super_features = []
        for bs in range(source_xzy.shape[0]):
            super_xyz = scatter_mean(source_xzy[bs], superpoint[bs], dim=0).unsqueeze(0).float()  # [1, super_num, 3]
            grouped_feature = self.super_grouper(points_xyz[bs].unsqueeze(0), super_xyz, mask_feats[bs].unsqueeze(0))  # [1, 288, super_num, nsample]
            super_feature = F.max_pool2d(grouped_feature, kernel_size=[1, grouped_feature.size(3)]).squeeze(-1).squeeze(0)  # [288, super_num]
            super_features.append(super_feature)

        # STEP 5. Query Points Generation
        end_points = self._generate_queries(
            points_xyz, points_features, end_points
        )
        cluster_feature = end_points['query_points_feature']    # (B, F=288, V=256)
        cluster_xyz = end_points['query_points_xyz']            # (B, V=256, 3)
        query = self.decoder_query_proj(cluster_feature)        
        query = query.transpose(1, 2).contiguous()              # (B, V=256, F=288)
        # projection 288 --> 64
        if self.contrastive_align_loss: 
            end_points['proposal_proj_queries'] = F.normalize(
                self.contrastive_align_projection_image(query), p=2, dim=-1
            )  # [B, 256, 64]

        # STEP 6.Proposals
        proposal_center, proposal_size = self.proposal_head(
            cluster_feature,
            base_xyz=cluster_xyz,
            end_points=end_points,
            prefix='proposal_'
        )
        base_xyz = proposal_center.detach().clone()
        base_size = proposal_size.detach().clone()
        query_mask = None
        query_last = None

        # STEP 7. Decoder
        for i in range(self.num_decoder_layers):
            prefix = 'last_' if i == self.num_decoder_layers-1 else f'{i}head_'

            # Position Embedding for Self-Attention
            if self.self_position_embedding == 'none':
                query_pos = None
            elif self.self_position_embedding == 'xyz_learned':
                query_pos = base_xyz
            elif self.self_position_embedding == 'loc_learned':
                query_pos = torch.cat([base_xyz, base_size], -1)
            else:
                raise NotImplementedError

            # step Transformer Decoder Layer
            query = self.decoder[i](
                query, points_features.transpose(1, 2).contiguous(),
                text_feats, query_pos,
                query_mask,
                text_padding_mask,
                detected_feats=(
                    detected_feats if self.butd
                    else None
                ),
                detected_mask=detected_mask if self.butd else None
            )  # (B, V, F)
            # step project
            if self.contrastive_align_loss:
                end_points[f'{prefix}proj_queries'] = F.normalize(
                    self.contrastive_align_projection_image(query), p=2, dim=-1
                )

            # step box Prediction head
            base_xyz, base_size = self.prediction_heads[i](
                query.transpose(1, 2).contiguous(),     # ([B, F=288, V=256])
                base_xyz=cluster_xyz,                   # ([B, 256, 3])
                end_points=end_points,  # 
                prefix=prefix
            )
            base_xyz = base_xyz.detach().clone()
            base_size = base_size.detach().clone()

            query_last = query

        # step Seg Prediction head
        query_last = self.x_query(query_last.transpose(1, 2)).transpose(1, 2)
        pred_masks = []
        for bs in range(query.shape[0]):
            pred_mask = self._seg_seeds_prediction(
                query_last[bs].unsqueeze(0),                                  # ([1, F=256, V=288])
                super_features[bs].unsqueeze(0),                             # ([1, F=288, V=super_num])
                end_points=end_points,  # 
                prefix=prefix
            ).squeeze(0)  
            pred_masks.append(pred_mask)

        end_points['last_pred_masks'] = pred_masks  # [B, 256, super_num]

        # debug
        # wordidx = np.array([
        #     0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 8, 9, 10, 11,
        #     12, 13, 13, 14, 15, 16, 16, 17, 17, 18, 18
        # ])  # 18+1（not mentioned）
        # tokenidx = np.array([
        #     1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19, 21, 23,
        #     25, 27, 29, 31, 32, 34, 36, 38, 39, 41, 42, 44, 45
        # ])  # 18 token span
        # proj_tokens = end_points['proj_tokens']  # (B, tokens, 64)
        # proj_queries = end_points['last_proj_queries']  # (B, Q, 64)
        # sem_scores = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2))
        # sem_scores_ = sem_scores / 0.07  # (B, Q, tokens)
        # sem_scores = torch.zeros(sem_scores_.size(0), sem_scores_.size(1), 256)
        # sem_scores = sem_scores.to(sem_scores_.device)
        # sem_scores[:, :sem_scores_.size(1), :sem_scores_.size(2)] = sem_scores_

        # sem_cls = torch.zeros_like(sem_scores)[..., :19] # ([B, 256, 19])
        # for w, t in zip(wordidx, tokenidx):
        #     sem_cls[..., w] += sem_scores[..., t]

        # class_id = sem_cls.argmax(-1)

        return end_points

    def init_bn_momentum(self):
        """Initialize batch-norm momentum."""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = 0.1
