# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class Matcher(nn.Module):
    def __init__(self, cost_class, cost_objectness, cost_giou, cost_center):
        """
        Parameters:
            cost_class:
        Returns:

        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_objectness = cost_objectness
        self.cost_giou = cost_giou
        self.cost_center = cost_center

    @torch.no_grad()
    def forward(self, outputs, targets):

        batchsize = outputs["last_sem_cls_scores"].shape[0]
        nqueries = outputs["last_sem_cls_scores"].shape[1]
        ngt = targets["sem_cls_label"].shape[1]
        nactual_gt = targets["nactual_gt"]

        # classification cost: batch x nqueries x ngt matrix
 
        # 18 class
        # pred_cls_prob = outputs["last_sem_cls_prob"].softmax(-1)  # [b, 256, 18(class)], there needs to use sem_cls_prob, not score
        # gt_box_sem_cls_labels = (
        #     targets["sem_cls_label"]
        #     .unsqueeze(1)
        #     .expand(batchsize, nqueries, ngt)
        # )
        # class_mat = -torch.gather(pred_cls_prob, 2, gt_box_sem_cls_labels)  # [b, 256, 132]

        # 256 class
        positive_map = targets["positive_map"]  # [B, 132, 256]
        pred_cls_prob = outputs["last_sem_cls_scores"].softmax(-1)  # [B, 132, 256]
        if pred_cls_prob.shape[-1] != positive_map.shape[-1]:
            positive_map = positive_map[..., :pred_cls_prob.shape[-1]]
        class_mat = torch.stack([
            -torch.matmul(pred_cls_prob[b], positive_map[b].transpose(0, 1))
            for b in range(batchsize)
        ], dim=0)  # [B, 256, 132]  targets["box_label_mask"][b].long()

        # giou cost: batch x nqueries x ngt
        giou_mat = -outputs["gious"].detach()

        final_cost = (
            self.cost_giou * giou_mat 
            # + self.cost_class * class_mat  # debug
        )

        final_cost = final_cost.detach().cpu().numpy()
        assignments = []

        # auxiliary variables useful for batched loss computation
        batch_size, nprop = final_cost.shape[0], final_cost.shape[1]
        per_prop_gt_inds = torch.zeros(
            [batch_size, nprop], dtype=torch.int64, device=pred_cls_prob.device
        )
        proposal_matched_mask = torch.zeros(
            [batch_size, nprop], dtype=torch.float32, device=pred_cls_prob.device
        )
        for b in range(batchsize):
            assign = []
            if nactual_gt[b] > 0:
                assign = linear_sum_assignment(final_cost[b, :, : nactual_gt[b]])
                assign = [
                    torch.from_numpy(x).long().to(device=pred_cls_prob.device)
                    for x in assign
                ]
                per_prop_gt_inds[b, assign[0]] = assign[1]
                proposal_matched_mask[b, assign[0]] = 1
            assignments.append(assign)

        return {
            "assignments": assignments,
            "per_prop_gt_inds": per_prop_gt_inds,
            "proposal_matched_mask": proposal_matched_mask,
        }

