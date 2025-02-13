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

from .backbone_ptv2maxpool import PMBEMBAttn
from .modules import (
    PointsObjClsModule, GeneralSamplingModule,
    ClsAgnosticPredictHead, PositionEmbeddingLearned
)
from .encoder_decoder_layers import (
    BiEncoder, BiEncoderLayer, BiDecoderLayer
)
from pointcept.models.builder import MODELS

@MODELS.register_module("eda_ptv2_dets3d")
class EDA_ptv2(nn.Module):
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
                 d_model=288, butd=False, pointnet_ckpt=None, 
                 data_path="/data/pointcloud/data_for_eda/scannet_others_processed/",
                 self_attend=True):
        """Initialize layers."""
        super().__init__()

        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.self_position_embedding = self_position_embedding
        self.contrastive_align_loss = contrastive_align_loss

        # Visual encoder
        self.backbone_net = PMBEMBAttn(in_channels=input_feature_dim)  # ptv2

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

        # Cross-encoder
        self.pos_embed = PositionEmbeddingLearned(3, d_model)
        bi_layer = BiEncoderLayer(
            d_model, dropout=0.1, activation="relu",
            n_heads=8, dim_feedforward=256,
            self_attend_lang=self_attend, self_attend_vis=self_attend,
            use_butd_enc_attn=False
        )
        self.cross_encoder = BiEncoder(bi_layer, 3)

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
                self_position_embedding=self_position_embedding, butd=False
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
        end_points = self.backbone_net(data_dict['point_clouds'], data_dict['offset'], data_dict['source_xzy'].shape[0])
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

        superpoint = data_dict['superpoint'] 
        end_points['superpoints'] = superpoint  # avoid bugs
        
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
                detected_feats=None,
                detected_mask=None
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

        return end_points

    def init_bn_momentum(self):
        """Initialize batch-norm momentum."""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = 0.1
