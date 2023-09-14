import torch.nn as nn

from pointcept.models.losses import build_criteria
from .builder import MODELS, build_model
from .losses.vqa_losses import HungarianMatcher, SetCriterion, compute_hungarian_loss
import numpy as np


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(self,
                 backbone=None,
                 criteria=None,
                 num_classes=40,
                 backbone_embed_dim=256,
                 ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_dict):
        feat = self.backbone(input_dict)
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)
        

@MODELS.register_module()
class DefaultGrounder(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        matcher = HungarianMatcher(1, 0, 2, True)
        losses = ['boxes', 'labels', 'contrastive_align', 'masks']
        self.set_criterion = SetCriterion(
                matcher=matcher,
                losses=losses, eos_coef=0.1, temperature=0.07)
        self.criterion = compute_hungarian_loss

    def forward(self, input_dict):

        inputs = {
            'point_clouds': input_dict['point_clouds'].float(), 
            'text': input_dict['utterances'],                   
            "det_boxes": input_dict['all_detected_boxes'],      
            "det_bbox_label_mask": input_dict['all_detected_bbox_label_mask'],  
            "det_class_ids": input_dict['all_detected_class_ids'],   
            "superpoint": input_dict['superpoint'], 
            "offset": input_dict['offset'],
            "source_xzy": input_dict['source_xzy']
        }
        end_points = self.backbone(inputs)
        end_points.update(input_dict)
        # train
        if self.training:
            loss, end_points = self.criterion(
            end_points, 6,
            self.set_criterion,
            query_points_obj_topk=5
        )
            return dict(loss=loss)
        # eval
        else:
            self.set_criterion.eval()
            return end_points