"""
Evaluate Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import numpy as np
import torch
import torch.distributed as dist
import pointops
from uuid import uuid4

import pointcept.utils.comm as comm
from pointcept.utils.misc import intersection_and_union_gpu
from pointcept.utils.grounding_evaluator import GroundingEvaluator as Gdeval
from pointcept.models.losses.vqa_losses import HungarianMatcher, SetCriterion, compute_hungarian_loss
from collections import OrderedDict, defaultdict
from pointcept.datasets.scanrefer_jointdc_v2c import SCANREFER  # for ScanRefer dataset
# from pointcept.datasets.nr3d_jointdc_v2c import SCANREFER  # for Nr3D dataset
from pointcept.utils.grounding_evaluator import _iou3d_par, box_cxcyczwhd_to_xyzxyz, box2points
import pointcept.utils.capeval.bleu.bleu as capblue
import pointcept.utils.capeval.cider.cider as capcider
import pointcept.utils.capeval.rouge.rouge as caprouge
import pointcept.utils.capeval.meteor.meteor as capmeteor
from pointcept.utils.proposal_parser import parse_predictions as parse_predictions_v2c
from pointcept.datasets.box_util import box3d_iou_batch_tensor
import wandb
import torch.nn.functional as F
from pointcept.datasets.preprocessing.scanrefer.model_util_scannet_v2c import ScannetDatasetConfig_V2C

from .default import HookBase
from .builder import HOOKS
import os, json
os.environ["TOKENIZERS_PARALLELISM"] = "false"
MAX_NUM_OBJ = 132


@HOOKS.register_module()
class ClsEvaluator(HookBase):
    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["cls_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            label = input_dict["category"]
            intersection, union, target = \
                intersection_and_union_gpu(
                    pred, label, self.trainer.cfg.data.num_classes, self.trainer.cfg.data.ignore_index)
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info("Test: [{iter}/{max_iter}] "
                                     "Loss {loss:.4f} ".format(iter=i + 1,
                                                               max_iter=len(self.trainer.val_loader),
                                                               loss=loss.item()))
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.trainer.logger.info("Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
            m_iou, m_acc, all_acc))
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info("Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                idx=i, name=self.trainer.cfg.data.names[i], iou=iou_class[i], accuracy=acc_class[i]))
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = all_acc  # save for saver
        self.trainer.comm_info["current_metric_name"] = "allAcc"  # save for saver

    def after_train(self):
        self.trainer.logger.info("Best {}: {:.4f}".format(
            "allAcc", self.trainer.best_metric_value))


@HOOKS.register_module()
class SemSegEvaluator(HookBase):
    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["seg_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            segment = input_dict["segment"]
            if "origin_coord" in input_dict.keys():
                idx, _ = pointops.knn_query(1, input_dict["coord"].float(), input_dict["offset"].int(),
                                            input_dict["origin_coord"].float(), input_dict["origin_offset"].int())
                pred = pred[idx.flatten().long()]
                segment = input_dict["origin_segment"]
            intersection, union, target = \
                intersection_and_union_gpu(
                    pred, segment, self.trainer.cfg.data.num_classes, self.trainer.cfg.data.ignore_index)
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            info = "Test: [{iter}/{max_iter}] ".format(iter=i + 1, max_iter=len(self.trainer.val_loader))
            if "origin_coord" in input_dict.keys():
                info = "Interp. " + info
            self.trainer.logger.info(info + "Loss {loss:.4f} ".format(iter=i + 1,
                                                                      max_iter=len(self.trainer.val_loader),
                                                                      loss=loss.item()))
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.trainer.logger.info("Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
            m_iou, m_acc, all_acc))
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info("Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                idx=i, name=self.trainer.cfg.data.names[i], iou=iou_class[i], accuracy=acc_class[i]))
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = m_iou  # save for saver
        self.trainer.comm_info["current_metric_name"] = "mIoU"  # save for saver

    def after_train(self):
        self.trainer.logger.info("Best {}: {:.4f}".format(
            "mIoU", self.trainer.best_metric_value))


@HOOKS.register_module()
class InsSegEvaluator(HookBase):
    def __init__(self,
                 segment_ignore_index=(-1,),
                 instance_ignore_index=-1):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

        self.valid_class_names = None  # update in before train
        self.overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        self.min_region_sizes = 100
        self.distance_threshes = float("inf")
        self.distance_confs = -float("inf")

    def before_train(self):
        self.valid_class_names = [self.trainer.cfg.data.names[i]
                                  for i in range(self.trainer.cfg.data.num_classes)
                                  if i not in self.segment_ignore_index]

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def associate_instances(self, pred, segment, instance):
        segment = segment.cpu().numpy()
        instance = instance.cpu().numpy()
        void_mask = np.in1d(segment, self.segment_ignore_index)

        assert pred["pred_classes"].shape[0] == pred["pred_scores"].shape[0] == pred["pred_masks"].shape[0]
        assert pred["pred_masks"].shape[1] == segment.shape[0] == instance.shape[0]
        # get gt instances
        gt_instances = dict()
        for i in range(self.trainer.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                gt_instances[self.trainer.cfg.data.names[i]] = []
        instance_ids, idx, counts = np.unique(instance, return_index=True, return_counts=True)
        segment_ids = segment[idx]
        for i in range(len(instance_ids)):
            if instance_ids[i] == self.instance_ignore_index:
                continue
            if segment_ids[i] in self.segment_ignore_index:
                continue
            gt_inst = dict()
            gt_inst["instance_id"] = instance_ids[i]
            gt_inst["segment_id"] = segment_ids[i]
            gt_inst["dist_conf"] = 0.
            gt_inst["med_dist"] = -1.
            gt_inst["vert_count"] = counts[i]
            gt_inst["matched_pred"] = []
            gt_instances[self.trainer.cfg.data.names[segment_ids[i]]].append(gt_inst)

        # get pred instances and associate with gt
        pred_instances = dict()
        for i in range(self.trainer.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                pred_instances[self.trainer.cfg.data.names[i]] = []
        instance_id = 0
        for i in range(len(pred["pred_classes"])):
            if pred["pred_classes"][i] in self.segment_ignore_index:
                continue
            pred_inst = dict()
            pred_inst["uuid"] = uuid4()
            pred_inst["instance_id"] = instance_id
            pred_inst["segment_id"] = pred["pred_classes"][i]
            pred_inst["confidence"] = pred["pred_scores"][i]
            pred_inst["mask"] = np.not_equal(pred["pred_masks"][i], 0)
            pred_inst["vert_count"] = np.count_nonzero(pred_inst["mask"])
            pred_inst["void_intersection"] = np.count_nonzero(np.logical_and(void_mask, pred_inst["mask"]))
            if pred_inst["vert_count"] < self.min_region_sizes:
                continue  # skip if empty
            segment_name = self.trainer.cfg.data.names[pred_inst["segment_id"]]
            matched_gt = []
            for gt_idx, gt_inst in enumerate(gt_instances[segment_name]):
                intersection = np.count_nonzero(np.logical_and(instance == gt_inst["instance_id"], pred_inst["mask"]))
                if intersection > 0:
                    gt_inst_ = gt_inst.copy()
                    pred_inst_ = pred_inst.copy()
                    gt_inst_["intersection"] = intersection
                    pred_inst_["intersection"] = intersection
                    matched_gt.append(gt_inst_)
                    gt_inst["matched_pred"].append(pred_inst_)
            pred_inst["matched_gt"] = matched_gt
            pred_instances[segment_name].append(pred_inst)
            instance_id += 1
        return gt_instances, pred_instances

    def evaluate_matches(self, scenes):
        overlaps = self.overlaps
        min_region_sizes = [self.min_region_sizes]
        dist_threshes = [self.distance_threshes]
        dist_confs = [self.distance_confs]

        # results: class x overlap
        ap_table = np.zeros((len(dist_threshes), len(self.valid_class_names), len(overlaps)), float)
        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
                zip(min_region_sizes, dist_threshes, dist_confs)):
            for oi, overlap_th in enumerate(overlaps):
                pred_visited = {}
                for scene in scenes:
                    for _ in scene["pred"]:
                        for label_name in self.valid_class_names:
                            for p in scene["pred"][label_name]:
                                if "uuid" in p:
                                    pred_visited[p["uuid"]] = False
                for li, label_name in enumerate(self.valid_class_names):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for scene in scenes:
                        pred_instances = scene["pred"][label_name]
                        gt_instances = scene["gt"][label_name]
                        # filter groups in ground truth
                        gt_instances = [gt for gt in gt_instances
                                        if gt["vert_count"] >= min_region_size
                                        and gt["med_dist"] <= distance_thresh
                                        and gt["dist_conf"] >= distance_conf]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                        cur_match = np.zeros(len(gt_instances), dtype=bool)
                        # collect matches
                        for (gti, gt) in enumerate(gt_instances):
                            found_match = False
                            for pred in gt["matched_pred"]:
                                # greedy assignments
                                if pred_visited[pred["uuid"]]:
                                    continue
                                overlap = float(pred["intersection"]) / (
                                        gt["vert_count"] + pred["vert_count"] - pred["intersection"])
                                if overlap > overlap_th:
                                    confidence = pred["confidence"]
                                    # if already have a prediction for this gt,
                                    # the prediction with the lower score is automatically a false positive
                                    if cur_match[gti]:
                                        max_score = max(cur_score[gti], confidence)
                                        min_score = min(cur_score[gti], confidence)
                                        cur_score[gti] = max_score
                                        # append false positive
                                        cur_true = np.append(cur_true, 0)
                                        cur_score = np.append(cur_score, min_score)
                                        cur_match = np.append(cur_match, True)
                                    # otherwise set score
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred["uuid"]] = True
                            if not found_match:
                                hard_false_negatives += 1
                        # remove non-matched ground truth instances
                        cur_true = cur_true[cur_match]
                        cur_score = cur_score[cur_match]

                        # collect non-matched predictions as false positive
                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred["matched_gt"]:
                                overlap = float(gt["intersection"]) / (
                                        gt["vert_count"] + pred["vert_count"] - gt["intersection"])
                                if overlap > overlap_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred["void_intersection"]
                                for gt in pred["matched_gt"]:
                                    if gt["segment_id"] in self.segment_ignore_index:
                                        num_ignore += gt["intersection"]
                                    # small ground truth instances
                                    if gt["vert_count"] < min_region_size or \
                                            gt["med_dist"] > distance_thresh or \
                                            gt["dist_conf"] < distance_conf:
                                        num_ignore += gt["intersection"]
                                proportion_ignore = float(num_ignore) / pred["vert_count"]
                                # if not ignored append false positive
                                if proportion_ignore <= overlap_th:
                                    cur_true = np.append(cur_true, 0)
                                    confidence = pred["confidence"]
                                    cur_score = np.append(cur_score, confidence)

                        # append to overall results
                        y_true = np.append(y_true, cur_true)
                        y_score = np.append(y_score, cur_score)

                    # compute average precision
                    if has_gt and has_pred:
                        # compute precision recall curve first

                        # sorting and cumsum
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        (thresholds, unique_indices) = np.unique(y_score_sorted, return_index=True)
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples = len(y_score_sorted)
                        # https://github.com/ScanNet/ScanNet/pull/26
                        # all predictions are non-matched but also all of them are ignored and not counted as FP
                        # y_true_sorted_cumsum is empty
                        # num_true_examples = y_true_sorted_cumsum[-1]
                        num_true_examples = y_true_sorted_cumsum[-1] if len(y_true_sorted_cumsum) > 0 else 0
                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        # deal with remaining
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p = float(tp) / (tp + fp)
                            r = float(tp) / (tp + fn)
                            precision[idx_res] = p
                            recall[idx_res] = r

                        # first point in curve is artificial
                        precision[-1] = 1.
                        recall[-1] = 0.

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.)

                        stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], "valid")
                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0
                    else:
                        ap_current = float("nan")
                    ap_table[di, li, oi] = ap_current
        d_inf = 0
        o50 = np.where(np.isclose(self.overlaps, 0.5))
        o25 = np.where(np.isclose(self.overlaps, 0.25))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.overlaps, 0.25)))
        ap_scores = dict()
        ap_scores["all_ap"] = np.nanmean(ap_table[d_inf, :, oAllBut25])
        ap_scores["all_ap_50%"] = np.nanmean(ap_table[d_inf, :, o50])
        ap_scores["all_ap_25%"] = np.nanmean(ap_table[d_inf, :, o25])
        ap_scores["classes"] = {}
        for (li, label_name) in enumerate(self.valid_class_names):
            ap_scores["classes"][label_name] = {}
            ap_scores["classes"][label_name]["ap"] = np.average(ap_table[d_inf, li, oAllBut25])
            ap_scores["classes"][label_name]["ap50%"] = np.average(ap_table[d_inf, li, o50])
            ap_scores["classes"][label_name]["ap25%"] = np.average(ap_table[d_inf, li, o25])
        return ap_scores

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        scenes = []
        for i, input_dict in enumerate(self.trainer.val_loader):
            assert len(input_dict["offset"]) == 1  # currently only support bs 1 for each GPU
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)

            loss = output_dict["loss"]

            segment = input_dict["segment"]
            instance = input_dict["instance"]
            # map to origin
            if "origin_coord" in input_dict.keys():
                idx, _ = pointops.knn_query(1, input_dict["coord"].float(), input_dict["offset"].int(),
                                            input_dict["origin_coord"].float(), input_dict["origin_offset"].int())
                idx = idx.cpu().flatten().long()
                output_dict["pred_masks"] = output_dict["pred_masks"][:, idx]
                segment = input_dict["origin_segment"]
                instance = input_dict["origin_instance"]

            gt_instances, pred_instance = self.associate_instances(output_dict, segment, instance)
            scenes.append(dict(gt=gt_instances, pred=pred_instance))

            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info("Test: [{iter}/{max_iter}] "
                                     "Loss {loss:.4f} ".format(iter=i + 1,
                                                               max_iter=len(self.trainer.val_loader),
                                                               loss=loss.item()))

        loss_avg = self.trainer.storage.history("val_loss").avg
        comm.synchronize()
        scenes_sync = comm.gather(scenes, dst=0)
        scenes = [scene for scenes_ in scenes_sync for scene in scenes_]
        ap_scores = self.evaluate_matches(scenes)
        all_ap = ap_scores["all_ap"]
        all_ap_50 = ap_scores["all_ap_50%"]
        all_ap_25 = ap_scores["all_ap_25%"]
        self.trainer.logger.info("Val result: mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f}.".format(
            all_ap, all_ap_50, all_ap_25))
        for (i, label_name) in enumerate(self.valid_class_names):
            ap = ap_scores["classes"][label_name]["ap"]
            ap_50 = ap_scores["classes"][label_name]["ap50%"]
            ap_25 = ap_scores["classes"][label_name]["ap25%"]
            self.trainer.logger.info("Class_{idx}-{name} Result: AP/AP50/AP25 {AP:.4f}/{AP50:.4f}/{AP25:.4f}".format(
                idx=i, name=label_name, AP=ap, AP50=ap_50, AP25=ap_25))
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mAP", all_ap, current_epoch)
            self.trainer.writer.add_scalar("val/AP50", all_ap_50, current_epoch)
            self.trainer.writer.add_scalar("val/AP25", all_ap_25, current_epoch)
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = all_ap_50  # save for saver
        self.trainer.comm_info["current_metric_name"] = "AP50"  # save for saver



@HOOKS.register_module()
class GroundingEvaluator(HookBase):

    def __init__(self, 
                 losses=['boxes', 'labels', 'contrastive_align', 'masks']):
        super().__init__()
        self.losses = losses

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            if (self.trainer.epoch + 1) % self.trainer.cfg.eval_freq == 0:
                self.eval()
            else:
                self.trainer.comm_info["current_metric_value"] = 0.0  # save for saver
                self.trainer.comm_info["current_metric_name"] = "REC_0.25IoU"  # save for saver

    def eval(self):
        self.trainer.logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
        self.trainer.model.eval()

        matcher = HungarianMatcher(1, 0, 2, True)
        set_criterion = SetCriterion(
                matcher=matcher,
                losses=self.losses, eos_coef=0.1, temperature=0.07)
        criterion = compute_hungarian_loss
        prefixes = ['last_', 'proposal_']
        prefixes += [f'{i}head_' for i in range(5)]
        evaluator = Gdeval(
            only_root=True, thresholds=[0.25, 0.5],     # TODO only_root=True
            topks=[1, 5, 10], prefixes=prefixes,
            filter_non_gt_boxes=False,
            logger=self.trainer.logger, losses=self.losses
        )
        for batch_idx, batch_data in enumerate(self.trainer.val_loader):
            # note forward and compute loss
            
            inputs = self._to_gpu(batch_data)

            if "train" not in inputs:
                inputs.update({"train": False})
            else:
                inputs["train"] = False

            # STEP Forward pass
            with torch.no_grad():
                end_points = self.trainer.model(inputs)

            # STEP Compute loss
            _, end_points = criterion(
                end_points, 6,
                set_criterion,
                query_points_obj_topk=5
            )

            for key in end_points:
                if 'pred_size' in key:
                    end_points[key] = torch.clamp(end_points[key], min=1e-6)

            # Accumulate statistics and print out
            stat_dict = {}
            stat_dict = self._accumulate_stats(stat_dict, end_points)
            if (batch_idx + 1) % 50 == 0:
                self.trainer.logger.info(f'Eval: [{batch_idx + 1}/{len(self.trainer.val_loader)}]  ')
                self.trainer.logger.info(''.join([
                    f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
                    for key in sorted(stat_dict.keys())
                    if 'loss' in key and 'proposal_' not in key
                    and 'last_' not in key and 'head_' not in key
                ]))

            if evaluator is not None:
                for prefix in prefixes:
                    # note only consider the last layer
                    if prefix != 'last_':
                        continue

                    # evaluation
                    evaluator.evaluate(end_points, prefix)  

            self.trainer.comm_info["current_metric_value"] = evaluator.get_best()  # save for saver
            self.trainer.comm_info["current_metric_name"] = "REC_0.25IoU"  # save for saver
            
        evaluator.print_stats()
        self.trainer.logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    def _accumulate_stats(self, stat_dict, end_points):
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                if isinstance(end_points[key], (float, int)):
                    stat_dict[key] += end_points[key]
                else:
                    stat_dict[key] += end_points[key].item()
        return stat_dict
    
    def _to_gpu(self, data_dict):
        if torch.cuda.is_available():
            for key in data_dict:
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda(non_blocking=True)
        return data_dict
    
    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("allAcc", self.trainer.best_metric_value)
        )


def prepare_corpus(raw_data, max_len: int=30) -> dict:
    # helper function to prepare ground truth captions
    corpus = defaultdict(list)
    object_id_to_name = defaultdict(lambda:'unknown')
    
    for data in raw_data:
        
        (scene_id, object_id, object_name) = data["scene_id"], data["object_id"], data["object_name"]
        
        # parse language tokens
        token = data["token"][:max_len]
        description = " ".join(["sos"] + token + ["eos"])
        key = f"{scene_id}|{object_id}|{object_name}"
        object_id_to_name[f"{scene_id}|{object_id}"] = object_name
        
        corpus[key].append(description)
        
    return corpus, object_id_to_name


def score_captions(corpus: dict, candidates: dict):
    
    bleu = capblue.Bleu(4).compute_score(corpus, candidates)
    cider = capcider.Cider().compute_score(corpus, candidates)
    rouge = caprouge.Rouge().compute_score(corpus, candidates)
    meteor = capmeteor.Meteor().compute_score(corpus, candidates)
    
    score_per_caption = {
        "bleu-1": [float(s) for s in bleu[1][0]],
        "bleu-2": [float(s) for s in bleu[1][1]],
        "bleu-3": [float(s) for s in bleu[1][2]],
        "bleu-4": [float(s) for s in bleu[1][3]],
        "cider": [float(s) for s in cider[1]],
        "rouge": [float(s) for s in rouge[1]],
        "meteor": [float(s) for s in meteor[1]],
    }
    
    message = '\n'.join([
        "[BLEU-1] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            bleu[0][0], max(bleu[1][0]), min(bleu[1][0])
        ),
        "[BLEU-2] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            bleu[0][1], max(bleu[1][1]), min(bleu[1][1])
        ),
        "[BLEU-3] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            bleu[0][2], max(bleu[1][2]), min(bleu[1][2])
        ),
        "[BLEU-4] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            bleu[0][3], max(bleu[1][3]), min(bleu[1][3])
        ),
        "[CIDEr] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            cider[0], max(cider[1]), min(cider[1])
        ),
        "[ROUGE-L] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            rouge[0], max(rouge[1]), min(rouge[1])
        ),
        "[METEOR] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            meteor[0], max(meteor[1]), min(meteor[1])
        )
    ])
    
    eval_metric = {
        "BLEU-4": bleu[0][3],
        "CiDEr": cider[0],
        "Rouge": rouge[0],
        "METEOR": meteor[0],
    }
    return score_per_caption, message, eval_metric

@HOOKS.register_module()
class CaptionEvaluator(HookBase):

    def __init__(self, 
                 losses=['boxes', 'labels', 'contrastive_align', 'captions']):
        super().__init__()
        self.test_min_iou = 0.50  # ability
        self.checkpoint_dir = "exp/captions_result"
        self.criterion = f'CiDEr@{self.test_min_iou}'
        dataset_config = ScannetDatasetConfig_V2C(18)
        # Used for AP calculation
        # self.config_dict = {
        #     'remove_empty_box': False, 'use_3d_nms': True,
        #     'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': True,
        #     'per_class_proposal': True, 'conf_thresh': 0.0,
        #     'dataset_config': dataset_config,
        #     'hungarian_loss': True
        # }  # EDA
        self.config_dict = {
            'remove_empty_box': True, 'use_3d_nms': True,
            'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': True,
            'per_class_proposal': True, 'conf_thresh': 0.05,
            'dataset_config': dataset_config,
            'hungarian_loss': True
        }  # V2C

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            if (self.trainer.epoch + 1) % self.trainer.cfg.eval_freq == 0:
                self.eval()
            else:
                self.trainer.comm_info["current_metric_value"] = 0.0  # save for saver
                self.trainer.comm_info["current_metric_name"] = "CiDEr@0.5"  # save for saver

    def eval(self):
        self.trainer.logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
        self.trainer.model.eval()
        if self.checkpoint_dir.split("/")[-1] == 'captions_result':
            self.checkpoint_dir = self.checkpoint_dir + "/" + self.trainer.cfg.save_path.split("/")[-1]
        
        # prepare ground truth caption labels
        print("preparing corpus...")
        corpus, object_id_to_name = prepare_corpus(
            SCANREFER['language']['val']
        )
        
        ### initialize and prepare for evaluation
        num_batches = len(self.trainer.val_loader)
        candidates = {'caption': OrderedDict({}), 'iou': defaultdict(float)}
        
        for curr_iter, batch_data in enumerate(self.trainer.val_loader):
            
            inputs = self._to_gpu(batch_data)

            with torch.no_grad():
                end_points = self.trainer.model(inputs)
            
            # match objects
            batch_size, nqueries, _ =  end_points['last_center'].shape
            gt_center = end_points['center_label'][:, :, 0:3]       
            gt_size = end_points['size_gts']                        
            gt_bboxes = torch.cat([gt_center, gt_size], dim=-1)
            pred_center = end_points['last_center']
            pred_size = end_points['last_pred_size']
            pred_bbox = torch.cat([pred_center, pred_size], dim=-1)
        
            match_box_ious = torch.stack([
                _iou3d_par(
                    box_cxcyczwhd_to_xyzxyz(gt_bboxes[b]),  # [B, 132, 6]
                    box_cxcyczwhd_to_xyzxyz(pred_bbox[b])  # [B, 256, 6] 
                )[0] for b in range(pred_bbox.shape[0])
            ], dim=0).transpose(-1, -2)  # batch, 256, 132
            
            match_box_ious, match_box_idxs = match_box_ious.max(-1) # batch, nqueries
            match_box_idxs = torch.gather(
                batch_data['gt_box_object_ids'], 1, 
                match_box_idxs
            ) # batch, nqueries

            # wordidx = np.array([
            #     0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 8, 9, 10, 11,
            #     12, 13, 13, 14, 15, 16, 16, 17, 17, 18, 18
            # ])  # 18+1（not mentioned）
            # tokenidx = np.array([
            #     1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19, 21, 23,
            #     25, 27, 29, 31, 32, 34, 36, 38, 39, 41, 42, 44, 45
            # ])  # 18 token span
            wordidx = np.array([ #len 27
                0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 8, 9, 10, 11,
                12, 13, 13, 14, 15, 16, 16, 17, 18, 18
            ])  # 18+1（not mentioned）
            tokenidx = np.array([ #len 27
                1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19, 21, 23,
                25, 27, 29, 31, 32, 34, 36, 38, 39, 41, 43, 44
            ])  # 18 token span
            # wordidx = np.array([ 
            #     2, 18, 18
            # ])
            # tokenidx = np.array([ 
            #     1, 3, 4
            # ])  # captions ability

            proj_tokens = end_points['proj_tokens']  # (B, tokens, 64)
            proj_queries = end_points['last_proj_queries']  # (B, Q, 64)
            sem_scores = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2))
            sem_scores_ = sem_scores / 0.07  # (B, Q, tokens)
            sem_scores = torch.zeros(sem_scores_.size(0), sem_scores_.size(1), 256)
            sem_scores = sem_scores.to(sem_scores_.device)
            sem_scores[:, :sem_scores_.size(1), :sem_scores_.size(2)] = sem_scores_
            end_points['last_sem_cls_scores'] = sem_scores  # ([B, 256, 256])

            sem_cls = torch.zeros_like(end_points['last_sem_cls_scores'])[..., :19] # ([B, 256, 19])
            for w, t in zip(wordidx, tokenidx):
                sem_cls[..., w] += end_points['last_sem_cls_scores'][..., t]
            end_points['last_sem_cls_scores'] = sem_cls
            sem_cls_prob = F.softmax(sem_cls, dim=-1)
            end_points['objectness_prob'] = 1 - sem_cls_prob[..., -1]  # [b, 256]

            # ---- Checkout bounding box ious and semantic logits
            good_bbox_masks = match_box_ious > self.test_min_iou     # batch, nqueries
            class_id = end_points["last_sem_cls_scores"].argmax(-1)
            good_bbox_masks &= end_points["last_sem_cls_scores"].argmax(-1) != (
                end_points["last_sem_cls_scores"].shape[-1] - 1
            )

            # ---- add nms to get accurate predictions, EDA
            _, nms_bbox_masks = parse_predictions(end_points, self.config_dict, "last_", size_cls_agnostic=True)  # [b, 256]

            # V2C  
            # box_corners = np.zeros((batch_size, nqueries, 8, 3))  # b, 256, 8, 3
            # pred_center_upright_camera = flip_axis_to_camera(pred_center.detach().cpu().numpy())
            # for i in range(batch_size):
            #     for j in range(nqueries):
            #         heading_angle = 0
            #         box_size = pred_size[i, j].detach().cpu().numpy()
            #         corners_3d_upright_camera = get_3d_box(box_size, heading_angle, pred_center_upright_camera[i, j, :])
            #         box_corners[i, j] = corners_3d_upright_camera
            # box_corners = torch.from_numpy(box_corners)  
            # nms_bbox_masks = parse_predictions_v2c(
            #     box_corners, 
            #     sem_cls_prob[..., :-1],  # [b, 256, 18]
            #     end_points['objectness_prob'],  # [b, 256]
            #     batch_data['point_clouds']
            # )  # [b, 256]

            nms_bbox_masks = torch.from_numpy(nms_bbox_masks).long() == 1
            good_bbox_masks &= nms_bbox_masks.to(good_bbox_masks.device)

            # vis
            # box_corners = torch.stack([torch.from_numpy(box2points(pred_bbox[b].cpu())) 
            #                            for b in range(batch_size)], dim=0).cuda()  # [b, 256, 8, 3]
            # gt_box_corners = torch.stack([torch.from_numpy(box2points(gt_bboxes[b].cpu())) 
            #                               for b in range(batch_size)], dim=0).cuda()  # [b, 132, 8, 3]
            # wandb.init(project="vis_s3d", name="dc_ability")
            # point_cloud_vis = batch_data['point_clouds'].reshape(-1, 50000, 6)[0].cpu()
            # og_color_vis = batch_data['og_color'][0].cpu()
            # point_cloud_vis[:, 3:] = (og_color_vis + torch.tensor([109.8, 97.2, 83.8]) / 256) * 256
            # gbc = gt_box_corners[0, batch_data['box_label_mask'][0].bool()].cpu()
            # # idx = sem_cls[0].argmax(0).cpu()
            # # bc = torch.stack([box_corners[0, id] for id in idx], dim=0)
            # # match_obj_id = torch.stack([match_box_idxs[0, id] for id in idx], dim=0)
            # bc = box_corners[0][good_bbox_masks[0].bool()]
            # obj_id = match_box_idxs[0][good_bbox_masks[0].bool()]
            # scene_id = str(SCANREFER['scene_list']['val'][batch_data["scan_idx"][0]])
            # color_list = [
            #     [255, 0, 0],
            #     [0, 255, 0],
            #     [0, 0, 255],
            #     [255, 255, 0],
            #     [255, 0, 255],
            #     [0, 255, 0],
            #     [128, 128, 96],
            #     [96, 128, 128],
            #     [128, 96, 128],
            #     [96, 128, 96],
            #     [54, 128, 256],
            #     [128, 54, 256],
            #     [128, 256, 54],
            #     [98, 158, 68],
            #     [64, 32, 45],
            #     [13, 68, 15],
            #     [31, 25, 66],
            #     [75, 85, 36],
            #     [136, 134, 212],
            #     [132, 61, 212],
            #     [34, 62, 230],
            #     [61, 36, 14],
            #     [38, 133, 45],
            #     [35, 121, 198],
            #     [35, 199, 244],
            #     [89, 135, 142],
            #     [123, 32, 121], 
            #     [34, 32, 12],
            #     [88, 76, 1],
            #     [121, 0, 33], 
            #     [67, 131, 36]
            # ]

            # wandb.log({
            #         "point_scene": wandb.Object3D({
            #             "type": "lidar/beta",
            #             "points": point_cloud_vis,
            #             "boxes": np.array(
            #                 # [
            #                 #     {
            #                 #         "corners": c.tolist(),
            #                 #         "label": "target",
            #                 #         "color": [0, 255, 0]
            #                 #     }
            #                 #     for c in gbc
            #                 # ] + 
            #                 [  # predicted boxes
            #                     {
            #                         "corners": c.tolist(),
            #                         "label": str(obj_id[i].item()),
            #                         "color": [255, 0, 0]
            #                     }
            #                     for i, c in enumerate(bc)
            #                 ]
            #             ),
            #         }),
            #     })
            
            good_bbox_masks = good_bbox_masks.cpu().tolist()
            
            captions = end_points["lang_cap"]  # batch, nqueries, [sentence]
            
            match_box_idxs = match_box_idxs.cpu().tolist()
            match_box_ious = match_box_ious.cpu().tolist()
            ### calculate measurable indicators on captions
            for idx, scene_id in enumerate(batch_data["scan_idx"].cpu().tolist()):
                scene_name = SCANREFER['scene_list']['val'][scene_id]
                for prop_id in range(nqueries):

                    if good_bbox_masks[idx][prop_id] is False:
                        continue
                    
                    match_obj_id = match_box_idxs[idx][prop_id]
                    match_obj_iou = match_box_ious[idx][prop_id]
                    
                    object_name = object_id_to_name[f"{scene_name}|{match_obj_id}"]
                    key = f"{scene_name}|{match_obj_id}|{object_name}"
                    
                    if match_obj_iou > candidates['iou'][key]:
                        candidates['iou'][key] = match_obj_iou
                        candidates['caption'][key] = [
                            captions[idx][prop_id]
                        ]
            
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            self.trainer.logger.info(
                f"Evaluate; Batch [{curr_iter + 1}/{num_batches}]; "
                f"Mem {mem_mb:0.2f}MB"
            )
        
        ### message out
        missing_proposals = len(corpus.keys() - candidates['caption'].keys())
        total_captions = len(corpus.keys())
        self.trainer.logger.info(
            f"\n----------------------Evaluation-----------------------\n"
            f"INFO: iou@{self.test_min_iou} matched proposals: "
            f"[{total_captions - missing_proposals} / {total_captions}], "
        )
        
        ### make up placeholders for undetected bounding boxes
        for missing_key in (corpus.keys() - candidates['caption'].keys()):
            candidates['caption'][missing_key] = ["sos eos"]
        
        # find annotated objects in scanrefer
        candidates = OrderedDict([
            (key, value) for key, value in sorted(candidates['caption'].items()) \
                if not key.endswith("unknown")
        ])
        score_per_caption, message, eval_metric = score_captions(
            OrderedDict([(key, corpus[key]) for key in candidates]), candidates
        )
        
        self.trainer.logger.info(message)
        
        with open(os.path.join(self.checkpoint_dir, "corpus_val.json"), "w") as f: 
            json.dump(corpus, f, indent=4)
        
        with open(os.path.join(self.checkpoint_dir, "pred_val.json"), "w") as f:
            json.dump(candidates, f, indent=4)
        
        with open(os.path.join(self.checkpoint_dir, "pred_gt_val.json"), "w") as f:
            pred_gt_val = {}
            for scene_object_id, scene_object_id_key in enumerate(candidates):
                pred_gt_val[scene_object_id_key] = {
                    'pred': candidates[scene_object_id_key],
                    'gt': corpus[scene_object_id_key],
                    'score': {
                        'bleu-1': score_per_caption['bleu-1'][scene_object_id],
                        'bleu-2': score_per_caption['bleu-2'][scene_object_id],
                        'bleu-3': score_per_caption['bleu-3'][scene_object_id],
                        'bleu-4': score_per_caption['bleu-4'][scene_object_id],
                        'CiDEr': score_per_caption['cider'][scene_object_id],
                        'rouge': score_per_caption['rouge'][scene_object_id],
                        'meteor': score_per_caption['meteor'][scene_object_id]
                    }
                }
            json.dump(pred_gt_val, f, indent=4)
        
        eval_metrics = {
            metric + f'@{self.test_min_iou}': score \
                for metric, score in eval_metric.items()
        }

        self.trainer.comm_info["current_metric_value"] = eval_metrics[self.criterion]  # save for saver
        self.trainer.comm_info["current_metric_name"] = "CiDEr@0.5"  # save for saver

        self.trainer.logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    def _accumulate_stats(self, stat_dict, end_points):
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                if isinstance(end_points[key], (float, int)):
                    stat_dict[key] += end_points[key]
                else:
                    stat_dict[key] += end_points[key].item()
        return stat_dict
    
    def _to_gpu(self, data_dict):
        if torch.cuda.is_available():
            for key in data_dict:
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda(non_blocking=True)
        return data_dict
    
    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("allAcc", self.trainer.best_metric_value)
        )



from tqdm import tqdm
from multiprocessing import Pool
from scipy.spatial import ConvexHull

def calc_iou(box_a, box_b):
    """Computes IoU of two axis aligned bboxes.
    Args:
        box_a, box_b: 6D of center and lengths        
    Returns:
        iou
    """

    max_a = box_a[0:3] + box_a[3:6] / 2
    max_b = box_b[0:3] + box_b[3:6] / 2
    min_max = np.array([max_a, max_b]).min(0)

    min_a = box_a[0:3] - box_a[3:6] / 2
    min_b = box_b[0:3] - box_b[3:6] / 2
    max_min = np.array([min_a, min_b]).max(0)
    if not ((min_max > max_min).all()):
        return 0.0

    intersection = (min_max - max_min).prod()
    vol_a = box_a[3:6].prod()
    vol_b = box_b[3:6].prod()
    union = vol_a + vol_b - intersection
    return 1.0 * intersection / union

def get_iou(bb1, bb2):
    iou3d = calc_iou(bb1, bb2)
    return iou3d

def poly_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def polygon_clip(subjectPolygon, clipPolygon):

    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return (outputList)

def convex_hull_intersection(p1, p2):
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0

def box3d_vol(corners):
    a = np.sqrt(np.sum((corners[0, :] - corners[1, :]) ** 2))
    b = np.sqrt(np.sum((corners[1, :] - corners[2, :]) ** 2))
    c = np.sqrt(np.sum((corners[0, :] - corners[4, :]) ** 2))
    return a * b * c

def box3d_iou(corners1, corners2):
    # corner points are in counter clockwise order
    rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
    rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]
    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area / (area1 + area2 - inter_area)
    ymax = min(corners1[0, 1], corners2[0, 1])
    ymin = max(corners1[4, 1], corners2[4, 1])
    inter_vol = inter_area * max(0.0, ymax - ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d

def get_iou_obb(bb1, bb2):
    iou3d, iou2d = box3d_iou(bb1, bb2)
    return iou3d

def get_iou_main(get_iou_func, args):
    return get_iou_func(*args)

def voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def eval_det_cls(pred, gt, ovthresh=0.25, use_07_metric=False, get_iou_func=get_iou):
    # construct gt objects
    class_recs = {}  # {img_id: {'bbox': bbox list, 'det': matched list}}
    npos = 0
    for img_id in gt.keys():
        bbox = np.array(gt[img_id])
        det = [False] * len(bbox)
        npos += len(bbox)
        class_recs[img_id] = {'bbox': bbox, 'det': det}
    # pad empty list to all other imgids
    for img_id in pred.keys():
        if img_id not in gt:
            class_recs[img_id] = {'bbox': np.array([]), 'det': []}
            
    # if npos==0:
    #     st()

    # construct dets
    image_ids = []
    confidence = []
    BB = []
    for img_id in pred.keys():
        for box, score in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            BB.append(box)
    confidence = np.array(confidence)
    BB = np.array(BB)  # (nd,4 or 8,3 or 6)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, ...]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        # if d%100==0: print(d)
        R = class_recs[image_ids[d]]
        bb = BB[d, ...].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            for j in range(BBGT.shape[0]):
                iou = get_iou_main(get_iou_func, (bb, BBGT[j, ...]))
                if iou > ovmax:
                    ovmax = iou
                    jmax = j

        # print d, ovmax
        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    # if npos==0:
    #     rec = np.zeros(tp.shape, dtype=np.float64)
    #     # print(tp.shape)
    # else:
    rec = tp / float(npos + 1e-8)

    # print('NPOS: ', npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

def eval_det_cls_wrapper(arguments):
    pred, gt, ovthresh, use_07_metric, get_iou_func = arguments
    rec, prec, ap = eval_det_cls(pred, gt, ovthresh, use_07_metric, get_iou_func)
    return (rec, prec, ap)

def softmax(x):
        """Numpy function for softmax."""
        shape = x.shape
        probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
        probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
        return probs

def flip_axis_to_camera(pc):
    """
    Flip X-right, Y-forward, Z-up to X-right, Y-down, Z-forward.

    Input and output are both (N, 3) array
    """
    pc2 = np.copy(pc)
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # cam X,Y,Z = depth X,-Z,Y
    pc2[..., 1] *= -1
    return pc2

def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                    [0, 1, 0],
                    [-s, 0, c]])

def get_3d_box(box_size, heading_angle, center):
    ''' box_size is array(l,w,h), heading_angle is radius clockwise from pos x axis, center is xyz of box center
        output (8,3) array for 3D box cornders
        Similar to utils/compute_orientation_3d
    '''
    R = roty(heading_angle)
    l, w, h = box_size
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2];
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2];
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2];
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + center[0];
    corners_3d[1, :] = corners_3d[1, :] + center[1];
    corners_3d[2, :] = corners_3d[2, :] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d

def flip_axis_to_depth(pc):
    """Inverse of flip_axis_to_camera."""
    pc2 = np.copy(pc)
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # depth X,Y,Z = cam X,Z,-Y
    pc2[..., 2] *= -1
    return pc2

def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0

def extract_pc_in_box3d( pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds

def nms_3d_faster_samecls(boxes, overlap_threshold, old_type=False):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    z1 = boxes[:, 2]
    x2 = boxes[:, 3]
    y2 = boxes[:, 4]
    z2 = boxes[:, 5]
    score = boxes[:, 6]
    cls = boxes[:, 7]
    area = (x2 - x1) * (y2 - y1) * (z2 - z1)

    I = np.argsort(score)
    pick = []
    while (I.size != 0):
        last = I.size
        i = I[-1]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[I[:last - 1]])
        yy1 = np.maximum(y1[i], y1[I[:last - 1]])
        zz1 = np.maximum(z1[i], z1[I[:last - 1]])
        xx2 = np.minimum(x2[i], x2[I[:last - 1]])
        yy2 = np.minimum(y2[i], y2[I[:last - 1]])
        zz2 = np.minimum(z2[i], z2[I[:last - 1]])
        cls1 = cls[i]
        cls2 = cls[I[:last - 1]]

        l = np.maximum(0, xx2 - xx1)
        w = np.maximum(0, yy2 - yy1)
        h = np.maximum(0, zz2 - zz1)

        if old_type:
            o = (l * w * h) / area[I[:last - 1]]
        else:
            inter = l * w * h
            o = inter / (area[i] + area[I[:last - 1]] - inter)
        o = o * (cls1 == cls2)

        I = np.delete(I, np.concatenate(([last - 1], np.where(o > overlap_threshold)[0])))

    return pick

def nms_3d_faster(boxes, overlap_threshold, old_type=False):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    z1 = boxes[:, 2]
    x2 = boxes[:, 3]
    y2 = boxes[:, 4]
    z2 = boxes[:, 5]
    score = boxes[:, 6]
    area = (x2 - x1) * (y2 - y1) * (z2 - z1)

    I = np.argsort(score)
    pick = []
    while (I.size != 0):
        last = I.size
        i = I[-1]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[I[:last - 1]])
        yy1 = np.maximum(y1[i], y1[I[:last - 1]])
        zz1 = np.maximum(z1[i], z1[I[:last - 1]])
        xx2 = np.minimum(x2[i], x2[I[:last - 1]])
        yy2 = np.minimum(y2[i], y2[I[:last - 1]])
        zz2 = np.minimum(z2[i], z2[I[:last - 1]])

        l = np.maximum(0, xx2 - xx1)
        w = np.maximum(0, yy2 - yy1)
        h = np.maximum(0, zz2 - zz1)

        if old_type:
            o = (l * w * h) / area[I[:last - 1]]
        else:
            inter = l * w * h
            o = inter / (area[i] + area[I[:last - 1]] - inter)

        I = np.delete(I, np.concatenate(([last - 1], np.where(o > overlap_threshold)[0])))

    return pick

def sigmoid(x):
    """Numpy function for sigmoid."""
    s = 1 / (1 + np.exp(-x))
    return s

def nms_2d_faster(boxes, overlap_threshold, old_type=False):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    score = boxes[:, 4]
    area = (x2 - x1) * (y2 - y1)

    I = np.argsort(score)
    pick = []
    while (I.size != 0):
        last = I.size
        i = I[-1]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[I[:last - 1]])
        yy1 = np.maximum(y1[i], y1[I[:last - 1]])
        xx2 = np.minimum(x2[i], x2[I[:last - 1]])
        yy2 = np.minimum(y2[i], y2[I[:last - 1]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        if old_type:
            o = (w * h) / area[I[:last - 1]]
        else:
            inter = w * h
            o = inter / (area[i] + area[I[:last - 1]] - inter)

        I = np.delete(I, np.concatenate(([last - 1], np.where(o > overlap_threshold)[0])))

    return pick

def parse_predictions(end_points, config_dict, prefix="", size_cls_agnostic=False):
    """ Parse predictions to OBB parameters and suppress overlapping boxes

    Args:
        end_points: dict
            {point_clouds, center, heading_scores, heading_residuals,
            size_scores, size_residuals, sem_cls_scores}
        config_dict: dict
            {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
            use_old_type_nms, conf_thresh, per_class_proposal}
    Returns:
        batch_pred_map_cls: a list of len == batch size (BS)
            [pred_list_i], i = 0, 1, ..., BS-1
            where pred_list_i = [(pred_sem_cls, box_params, box_score)_j]
            where j = 0, ..., num of valid detections - 1 from sample input i
    """
    pred_center = end_points[f'{prefix}center']  # (B,num_proposal=256,3)
    # pred_heading_class = torch.argmax(end_points[f'{prefix}heading_scores'], -1)  # B,num_proposal
    # pred_heading_residual = torch.gather(end_points[f'{prefix}heading_residuals'], 2,
    #                                      pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
    # pred_heading_residual.squeeze_(2)

    if size_cls_agnostic:
        pred_size = end_points[f'{prefix}pred_size']  # (B, num_proposal, 3)
    else:
        pred_size_class = torch.argmax(end_points[f'{prefix}size_scores'], -1)  # B,num_proposal
        pred_size_residual = torch.gather(end_points[f'{prefix}size_residuals'], 2,
                                        pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1,
                                                                                            3))  # B,num_proposal,1,3
        pred_size_residual.squeeze_(2)
    pred_sem_cls = torch.argmax(end_points[f'{prefix}sem_cls_scores'][..., :-1], -1)  # B,num_proposal
    sem_cls_probs = softmax(end_points[f'{prefix}sem_cls_scores'].detach().cpu().numpy())  # softmax, B,num_proposal,19

    num_proposal = pred_center.shape[1]     # 256
    # Since we operate in upright_depth coord for points, while util functions
    # assume upright_camera coord.
    # pred_size_check = end_points[f'{prefix}pred_size']  # B,num_proposal,3
    # pred_bbox_check = end_points[f'{prefix}bbox_check']  # B,num_proposal,3

    bsize = pred_center.shape[0]
    pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3))  # b, 256, 8, 3
    pred_center_upright_camera = flip_axis_to_camera(pred_center.detach().cpu().numpy())
    for i in range(bsize):
        for j in range(num_proposal):
            heading_angle = 0  #config_dict['dataset_config'].class2angle( \
            #     pred_heading_class[i, j].detach().cpu().numpy(), pred_heading_residual[i, j].detach().cpu().numpy())
            if size_cls_agnostic:
                box_size = pred_size[i, j].detach().cpu().numpy()
            else:
                box_size = config_dict['dataset_config'].class2size( \
                    int(pred_size_class[i, j].detach().cpu().numpy()), pred_size_residual[i, j].detach().cpu().numpy())
            
            corners_3d_upright_camera = get_3d_box(box_size, heading_angle, pred_center_upright_camera[i, j, :])
            pred_corners_3d_upright_camera[i, j] = corners_3d_upright_camera

    K = pred_center.shape[1]  # K==num_proposal
    nonempty_box_mask = np.ones((bsize, K))

    if config_dict['remove_empty_box']:
        # -------------------------------------
        # Remove predicted boxes without any point within them..
        batch_pc = end_points['point_clouds'].reshape(bsize, -1, 6).cpu().numpy()[:, :, 0:3]  # B,N,3
        for i in range(bsize):
            pc = batch_pc[i, :, :]  # (N,3)
            for j in range(K):
                box3d = pred_corners_3d_upright_camera[i, j, :, :]  # (8,3)
                box3d = flip_axis_to_depth(box3d)
                pc_in_box, inds = extract_pc_in_box3d(pc, box3d)
                if len(pc_in_box) < 5:
                    nonempty_box_mask[i, j] = 0
        # -------------------------------------
    if config_dict.get('hungarian_loss', False):
        # obj_logits = np.zeros(pred_center[:,:,None,0].shape) + 5 # (B,K,1)
        # obj_logits[end_points[f'{prefix}indices']] = 5
        if f'{prefix}objectness_scores' in end_points:
            obj_logits = end_points[f'{prefix}objectness_scores'].detach().cpu().numpy()
            obj_prob = sigmoid(obj_logits)  # (B,K)
        else: 
            obj_prob = (1 - sem_cls_probs[:,:,-1])
            sem_cls_probs = sem_cls_probs[..., :-1] / obj_prob[..., None]  # differences
    else:
        obj_logits = end_points[f'{prefix}objectness_scores'].detach().cpu().numpy()
        obj_prob = sigmoid(obj_logits)[:, :, 0]  # (B,256)
    
    if not config_dict['use_3d_nms']:
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_2d_with_prob = np.zeros((K, 5))
            for j in range(K):
                boxes_2d_with_prob[j, 0] = np.min(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_2d_with_prob[j, 2] = np.max(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_2d_with_prob[j, 1] = np.min(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_2d_with_prob[j, 3] = np.max(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_2d_with_prob[j, 4] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_2d_faster(boxes_2d_with_prob[nonempty_box_mask[i, :] == 1, :],
                                config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert (len(pick) > 0)
            pred_mask[i, nonempty_box_inds[pick]] = 1
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict['use_3d_nms'] and (not config_dict['cls_nms']):
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 7))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 1] = np.min(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 2] = np.min(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 3] = np.max(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 4] = np.max(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 5] = np.max(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_3d_faster(boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                                config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert (len(pick) > 0)
            pred_mask[i, nonempty_box_inds[pick]] = 1
        # ---------- NMS output: pred_mask in (B,K) -----------
    # 3D NMS
    elif config_dict['use_3d_nms'] and config_dict['cls_nms']:
        # ---------- NMS input: pred_with_prob in (B,K,8) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 8))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 1] = np.min(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 2] = np.min(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 3] = np.max(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 4] = np.max(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 5] = np.max(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
                boxes_3d_with_prob[j, 7] = pred_sem_cls[i, j]  # only suppress if the two boxes are of the same class!!
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_3d_faster_samecls(boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                                        config_dict['nms_iou'], config_dict['use_old_type_nms'])
            # assert (len(pick) > 0)
            if len(pick) > 0:
                pred_mask[i, nonempty_box_inds[pick]] = 1
        end_points[f'{prefix}pred_mask'] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------

    batch_pred_map_cls = []  # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
    for i in range(bsize):
        if config_dict['per_class_proposal']:
            cur_list = []
            for ii in range(config_dict['dataset_config'].num_class):
                # if config_dict.get('hungarian_loss', False) and ii == config_dict['dataset_config'].num_class - 1:
                #    continue
                try:
                    cur_list += [
                        (ii, pred_corners_3d_upright_camera[i, j], sem_cls_probs[i, j, ii] * obj_prob[i, j])
                        for j in range(pred_center.shape[1])
                        if pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict['conf_thresh']
                    ]
                except:
                    st()
            batch_pred_map_cls.append(cur_list)
        else:
            batch_pred_map_cls.append([(pred_sem_cls[i, j].item(), pred_corners_3d_upright_camera[i, j], obj_prob[i, j]) \
                                    for j in range(pred_center.shape[1]) if
                                    pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict['conf_thresh']])

    return batch_pred_map_cls, pred_mask

class ScannetDatasetConfig:

    def __init__(self, num_class=485, agnostic=False):
        self.num_class = num_class if not agnostic else 1  # 18
        self.num_heading_bin = 1
        self.num_size_cluster = num_class
        if num_class == 18:
            self.type2class = {'cabinet': 0, 'bed': 1, 'chair': 2, 'couch': 3, 'table': 4, 'door': 5,
                               'window': 6, 'bookshelf': 7, 'picture': 8, 'counter': 9, 'desk': 10, 'curtain': 11,
                               'refrigerator': 12, 'shower curtain': 13, 'toilet': 14, 'sink': 15, 'bathtub': 16,
                               'other furniture': 17}
        else:
            self.type2class = {'wall': 0, 'chair': 1, 'floor': 2, 'table': 3, 'door': 4, 'couch': 5, 'cabinet': 6, 'shelf': 7, 'desk': 8, 'office chair': 9, 'bed': 10, 'pillow': 11, 'sink': 12, 'picture': 13, 'window': 14, 'toilet': 15, 'bookshelf': 16, 'monitor': 17, 'curtain': 18, 'book': 19, 'armchair': 20, 'coffee table': 21, 'drawer': 22, 'box': 23, 'refrigerator': 24, 'lamp': 25, 'kitchen cabinet': 26, 'towel': 27, 'clothes': 28, 'tv': 29, 'nightstand': 30, 'counter': 31, 'dresser': 32, 'stool': 33, 'couch cushions': 34, 'plant': 35, 'ceiling': 36, 'bathtub': 37, 'end table': 38, 'dining table': 39, 'keyboard': 40, 'bag': 41, 'backpack': 42, 'toilet paper': 43, 'printer': 44, 'tv stand': 45, 'whiteboard': 46, 'carpet': 47, 'blanket': 48, 'shower curtain': 49, 'trash can': 50, 'closet': 51, 'staircase': 52, 'microwave': 53, 'rug': 54, 'stove': 55, 'shoe': 56, 'computer tower': 57, 'bottle': 58, 'bin': 59, 'ottoman': 60, 'bench': 61, 'board': 62, 'washing machine': 63, 'mirror': 64, 'copier': 65, 'basket': 66, 'sofa chair': 67, 'file cabinet': 68, 'fan': 69, 'laptop': 70, 'shower': 71, 'paper': 72, 'person': 73, 'headboard': 74, 'paper towel dispenser': 75, 'faucet': 76, 'oven': 77, 'footstool': 78, 'blinds': 79, 'rack': 80, 'plate': 81, 'blackboard': 82, 'piano': 83, 'heater': 84, 'soap': 85, 'suitcase': 86, 'rail': 87, 'radiator': 88, 'recycling bin': 89, 'container': 90, 'closet wardrobe': 91, 'soap dispenser': 92, 'telephone': 93, 'bucket': 94, 'clock': 95, 'stand': 96, 'light': 97, 'laundry basket': 98, 'pipe': 99, 'round table': 100, 'clothes dryer': 101, 'coat': 102, 'guitar': 103, 'toilet paper holder': 104, 'seat': 105, 'step': 106, 'speaker': 107, 'vending machine': 108, 'column': 109, 'bicycle': 110, 'ladder': 111, 'cover': 112, 'bathroom stall': 113, 'foosball table': 114, 'shower wall': 115, 'chest': 116, 'cup': 117, 'jacket': 118, 'storage bin': 119, 'screen': 120, 'coffee maker': 121, 'hamper': 122, 'dishwasher': 123, 'paper towel roll': 124, 'machine': 125, 'mat': 126, 'windowsill': 127, 'tap': 128, 'pool table': 129, 'hand dryer': 130, 'bar': 131, 'frame': 132, 'toaster': 133, 'handrail': 134, 'bulletin board': 135, 'ironing board': 136, 'fireplace': 137, 'soap dish': 138, 'kitchen counter': 139, 'glass': 140, 'doorframe': 141, 'toilet paper dispenser': 142, 'mini fridge': 143, 'fire extinguisher': 144, 'shampoo bottle': 145, 'ball': 146, 'hat': 147, 'shower curtain rod': 148, 'toiletry': 149, 'water cooler': 150, 'desk lamp': 151, 'paper cutter': 152, 'switch': 153, 'tray': 154, 'shower door': 155, 'shirt': 156, 'pillar': 157, 'ledge': 158, 'vase': 159, 'toaster oven': 160, 'mouse': 161, 'nerf gun': 162, 'toilet seat cover dispenser': 163, 'can': 164, 'furniture': 165, 'cart': 166, 'step stool': 167, 'dispenser': 168, 'storage container': 169, 'side table': 170, 'lotion': 171, 'cooking pot': 172, 'toilet brush': 173, 'scale': 174, 'tissue box': 175, 'remote': 176, 'light switch': 177, 'crate': 178, 'ping pong table': 179, 'platform': 180, 'slipper': 181, 'power outlet': 182, 'cutting board': 183, 'controller': 184, 'decoration': 185, 'trolley': 186, 'sign': 187, 'projector': 188, 'sweater': 189, 'globe': 190, 'closet door': 191, 'plastic container': 192, 'statue': 193, 'vacuum cleaner': 194, 'wet floor sign': 195, 'candle': 196, 'easel': 197, 'wall hanging': 198, 'dumbell': 199, 'ping pong paddle': 200, 'plunger': 201, 'soap bar': 202, 'stuffed animal': 203, 'water fountain': 204, 'footrest': 205, 'headphones': 206, 'plastic bin': 207, 'coatrack': 208, 'dish rack': 209, 'broom': 210, 'guitar case': 211, 'mop': 212, 'magazine': 213, 'range hood': 214, 'scanner': 215, 'bathrobe': 216, 'futon': 217, 'dustpan': 218, 'hand towel': 219, 'organizer': 220, 'map': 221, 'helmet': 222, 'hair dryer': 223, 'exercise ball': 224, 'iron': 225, 'studio light': 226, 'cabinet door': 227, 'exercise machine': 228, 'workbench': 229, 'water bottle': 230, 'handicap bar': 231, 'tank': 232, 'purse': 233, 'vent': 234, 'piano bench': 235, 'bunk bed': 236, 'shoe rack': 237, 'shower floor': 238, 'case': 239, 'swiffer': 240, 'stapler': 241, 'cable': 242, 'garbage bag': 243, 'banister': 244, 'trunk': 245, 'tire': 246, 'folder': 247, 'car': 248, 'flower stand': 249, 'water pitcher': 250, 'loft bed': 251, 'shopping bag': 252, 'curtain rod': 253, 'alarm': 254, 'washcloth': 255, 'toolbox': 256, 'sewing machine': 257, 'mailbox': 258, 'toothpaste': 259, 'rope': 260, 'electric panel': 261, 'bowl': 262, 'boiler': 263, 'paper bag': 264, 'alarm clock': 265, 'music stand': 266, 'instrument case': 267, 'paper tray': 268, 'paper shredder': 269, 'projector screen': 270, 'boots': 271, 'kettle': 272, 'mail tray': 273, 'cat litter box': 274, 'covered box': 275, 'ceiling fan': 276, 'cardboard': 277, 'binder': 278, 'beachball': 279, 'envelope': 280, 'thermos': 281, 'breakfast bar': 282, 'dress rack': 283, 'frying pan': 284, 'divider': 285, 'rod': 286, 'magazine rack': 287, 'laundry detergent': 288, 'sofa bed': 289, 'storage shelf': 290, 'loofa': 291, 'bycicle': 292, 'file organizer': 293, 'fire hose': 294, 'media center': 295, 'umbrella': 296, 'barrier': 297, 'subwoofer': 298, 'stepladder': 299, 'shorts': 300, 'rocking chair': 301, 'elliptical machine': 302, 'coffee mug': 303, 'jar': 304, 'door wall': 305, 'traffic cone': 306, 'pants': 307, 'garage door': 308, 'teapot': 309, 'barricade': 310, 'exit sign': 311, 'canopy': 312, 'kinect': 313, 'kitchen island': 314, 'messenger bag': 315, 'buddha': 316, 'block': 317, 'stepstool': 318, 'tripod': 319, 'chandelier': 320, 'smoke detector': 321, 'baseball cap': 322, 'toothbrush': 323, 'bathroom counter': 324, 'object': 325, 'bathroom vanity': 326, 'closet wall': 327, 'laundry hamper': 328, 'bathroom stall door': 329, 'ceiling light': 330, 'trash bin': 331, 'dumbbell': 332, 'stair rail': 333, 'tube': 334, 'bathroom cabinet': 335, 'cd case': 336, 'closet rod': 337, 'coffee kettle': 338, 'wardrobe cabinet': 339, 'structure': 340, 'shower head': 341, 'keyboard piano': 342, 'case of water bottles': 343, 'coat rack': 344, 'storage organizer': 345, 'folded chair': 346, 'fire alarm': 347, 'power strip': 348, 'calendar': 349, 'poster': 350, 'potted plant': 351, 'luggage': 352, 'mattress': 353, 'hand rail': 354, 'folded table': 355, 'poster tube': 356, 'thermostat': 357, 'flip flops': 358, 'cloth': 359, 'banner': 360, 'clothes hanger': 361, 'whiteboard eraser': 362, 'shower control valve': 363, 'compost bin': 364, 'teddy bear': 365, 'pantry wall': 366, 'tupperware': 367, 'beer bottles': 368, 'salt': 369, 'mirror doors': 370, 'folded ladder': 371, 'carton': 372, 'soda stream': 373, 'metronome': 374, 'music book': 375, 'rice cooker': 376, 'dart board': 377, 'grab bar': 378, 'flowerpot': 379, 'painting': 380, 'railing': 381, 'stair': 382, 'quadcopter': 383, 'pitcher': 384, 'hanging': 385, 'mail': 386, 'closet ceiling': 387, 'hoverboard': 388, 'beanbag chair': 389, 'spray bottle': 390, 'soap bottle': 391, 'ikea bag': 392, 'duffel bag': 393, 'oven mitt': 394, 'pot': 395, 'hair brush': 396, 'tennis racket': 397, 'display case': 398, 'bananas': 399, 'carseat': 400, 'coffee box': 401, 'clothing rack': 402, 'bath walls': 403, 'podium': 404, 'storage box': 405, 'dolly': 406, 'shampoo': 407, 'changing station': 408, 'crutches': 409, 'grocery bag': 410, 'pizza box': 411, 'shaving cream': 412, 'luggage rack': 413, 'urinal': 414, 'hose': 415, 'bike pump': 416, 'bear': 417, 'humidifier': 418, 'mouthwash bottle': 419, 'golf bag': 420, 'food container': 421, 'card': 422, 'mug': 423, 'boxes of paper': 424, 'flag': 425, 'rolled poster': 426, 'wheel': 427, 'blackboard eraser': 428, 'doll': 429, 'laundry bag': 430, 'sponge': 431, 'lotion bottle': 432, 'lunch box': 433, 'sliding wood door': 434, 'briefcase': 435, 'bath products': 436, 'star': 437, 'coffee bean bag': 438, 'ipad': 439, 'display rack': 440, 'massage chair': 441, 'paper organizer': 442, 'cap': 443, 'dumbbell plates': 444, 'elevator': 445, 'cooking pan': 446, 'trash bag': 447, 'santa': 448, 'jewelry box': 449, 'boat': 450, 'sock': 451, 'plastic storage bin': 452, 'dishwashing soap bottle': 453, 'xbox controller': 454, 'airplane': 455, 'conditioner bottle': 456, 'tea kettle': 457, 'wall mounted coat rack': 458, 'film light': 459, 'sofa': 460, 'pantry shelf': 461, 'fish': 462, 'toy dinosaur': 463, 'cone': 464, 'fire sprinkler': 465, 'contact lens solution bottle': 466, 'hand sanitzer dispenser': 467, 'pen holder': 468, 'wig': 469, 'night light': 470, 'notepad': 471, 'drum set': 472, 'closet shelf': 473, 'exercise bike': 474, 'soda can': 475, 'stovetop': 476, 'telescope': 477, 'battery disposal jar': 478, 'closet floor': 479, 'clip': 480, 'display': 481, 'postcard': 482, 'paper towel': 483, 'food bag': 484}

        self.class2type = {self.type2class[t]: t for t in self.type2class}
        if num_class == 18:
            self.nyu40ids = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
        else:
            self.nyu40ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 110, 111, 112, 115, 116, 117, 118, 119, 120, 121, 122, 123, 125, 126, 128, 129, 130, 131, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 143, 144, 145, 146, 148, 152, 154, 155, 156, 157, 159, 160, 161, 163, 165, 166, 167, 168, 169, 170, 174, 177, 179, 180, 182, 185, 188, 189, 191, 193, 194, 195, 202, 204, 208, 212, 213, 214, 216, 220, 221, 222, 225, 226, 228, 229, 230, 231, 232, 233, 234, 235, 238, 242, 245, 247, 250, 257, 261, 264, 265, 269, 276, 280, 281, 283, 284, 286, 289, 291, 297, 298, 300, 301, 304, 305, 307, 312, 316, 319, 323, 325, 331, 332, 339, 342, 345, 346, 354, 356, 357, 361, 365, 366, 370, 372, 378, 379, 385, 386, 389, 392, 395, 397, 399, 408, 410, 411, 415, 417, 432, 434, 435, 436, 440, 448, 450, 452, 459, 461, 484, 488, 494, 506, 513, 518, 523, 525, 529, 540, 546, 556, 561, 562, 563, 570, 572, 581, 591, 592, 599, 609, 612, 621, 643, 657, 673, 682, 689, 693, 712, 719, 726, 730, 733, 746, 748, 750, 765, 776, 786, 794, 801, 803, 813, 814, 815, 816, 817, 819, 851, 857, 885, 893, 907, 919, 947, 948, 955, 976, 997, 1005, 1009, 1028, 1051, 1063, 1072, 1083, 1098, 1116, 1117, 1122, 1125, 1126, 1135, 1156, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1232, 1233, 1234, 1235, 1236, 1237, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1250, 1252, 1253, 1255, 1256, 1257, 1258, 1259, 1260, 1261, 1262, 1264, 1265, 1268, 1269, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1279, 1280, 1282, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295, 1296, 1297, 1298, 1299, 1300, 1301, 1302, 1304, 1305, 1307, 1308, 1309, 1311, 1312, 1313, 1316, 1318, 1319, 1320, 1321, 1324, 1326, 1327, 1329, 1330, 1331, 1334, 1335, 1337, 1339, 1340, 1344, 1346, 1347, 1350, 1351, 1352, 1353, 1356])
        self.nyu40id2class = {nyu40id: i for i, nyu40id in enumerate(list(self.nyu40ids))}


class APCalculator(object):
    ''' Calculating Average Precision '''

    def __init__(self, ap_iou_thresh=0.25, class2type_map=None):
        """
        Args:
            ap_iou_thresh: float between 0 and 1.0
                IoU threshold to judge whether a prediction is positive.
            class2type_map: [optional] dict {class_int:class_name}
        """
        self.ap_iou_thresh = ap_iou_thresh
        self.class2type_map = class2type_map
        self.uniq_gt_classes = set()
        self.reset()

    def step(self, batch_pred_map_cls, batch_gt_map_cls):
        """ Accumulate one batch of prediction and groundtruth.
        
        Args:
            batch_pred_map_cls: a list of lists [[(pred_cls, pred_box_params, score),...],...]
            batch_gt_map_cls: a list of lists [[(gt_cls, gt_box_params),...],...]
                should have the same length with batch_pred_map_cls (batch_size)
        """

        bsize = len(batch_pred_map_cls)
        assert (bsize == len(batch_gt_map_cls))
        for i in range(bsize):
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i]
            for classname, _ in batch_gt_map_cls[i]:
                self.uniq_gt_classes.add(classname)
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i]
            self.scan_cnt += 1

    def _volume_par(self, box):
        return (
            (box[:, 3] - box[:, 0])
            * (box[:, 4] - box[:, 1])
            * (box[:, 5] - box[:, 2])
        )

    def _intersect_par(self, box_a, box_b):
        xA = torch.max(box_a[:, 0][:, None], box_b[:, 0][None, :])
        yA = torch.max(box_a[:, 1][:, None], box_b[:, 1][None, :])
        zA = torch.max(box_a[:, 2][:, None], box_b[:, 2][None, :])
        xB = torch.min(box_a[:, 3][:, None], box_b[:, 3][None, :])
        yB = torch.min(box_a[:, 4][:, None], box_b[:, 4][None, :])
        zB = torch.min(box_a[:, 5][:, None], box_b[:, 5][None, :])
        return (
            torch.clamp(xB - xA, 0)
            * torch.clamp(yB - yA, 0)
            * torch.clamp(zB - zA, 0)
        )

    def corners_to_ends(self, box):
        min_xyz = torch.min(box, axis=0)[0]
        max_xyz = torch.max(box, axis=0)[0]
        return torch.cat((min_xyz, max_xyz))

    def _iou3d_par(self, box_a, box_b):
        intersection = self._intersect_par(box_a, box_b)
        vol_a = self._volume_par(box_a)
        vol_b = self._volume_par(box_b)
        union = vol_a[:, None] + vol_b[None, :] - intersection
        return intersection / union, union

    def generalized_box_iou3d(self, boxes1, boxes2):
        """
        Generalized IoU from https://giou.stanford.edu/

        The boxes should be in [x0, y0, x1, y1] format
        Returns a [N, M] pairwise matrix, where N = len(boxes1)
        and M = len(boxes2)
        """
        # degenerate boxes gives inf / nan results
        # so do an early check

        assert (boxes1[:, 3:] >= boxes1[:, :3]).all()
        assert (boxes2[:, 3:] >= boxes2[:, :3]).all()
        iou, union = self._iou3d_par(boxes1, boxes2)

        lt = torch.min(boxes1[:, None, :3], boxes2[:, :3])
        rb = torch.max(boxes1[:, None, 3:], boxes2[:, 3:])

        wh = (rb - lt).clamp(min=0)  # [N,M,3]
        volume = wh[:, :, 0] * wh[:, :, 1] * wh[:, :, 2]

        return iou - (volume - union) / volume

    def eval_det_multiprocessing(self, pred_all, gt_all, ovthresh=0.25, use_07_metric=False, get_iou_func=get_iou):
        """ Generic functions to compute precision/recall for object detection
            for multiple classes.
            Input:
                pred_all: map of {img_id: [(classname, bbox, score)]}
                gt_all: map of {img_id: [(classname, bbox)]}
                ovthresh: scalar, iou threshold
                use_07_metric: bool, if true use VOC07 11 point method
            Output:
                rec: {classname: rec}
                prec: {classname: prec_all}
                ap: {classname: scalar}
        """
        pred = {}  # map {classname: pred}
        gt = {}  # map {classname: gt}
        for img_id in pred_all.keys():
            for classname, bbox, score in pred_all[img_id]:
                # if classname not in VALID_TEST_CLASSES:
                #     continue
                if classname not in pred: pred[classname] = {}
                if img_id not in pred[classname]:
                    pred[classname][img_id] = []
                if classname not in gt: gt[classname] = {}
                if img_id not in gt[classname]:
                    gt[classname][img_id] = []
                pred[classname][img_id].append((bbox, score))
        for img_id in gt_all.keys():
            for classname, bbox in gt_all[img_id]:
                # if classname not in VALID_TEST_CLASSES:
                #     continue
                if classname not in gt: gt[classname] = {}
                if img_id not in gt[classname]:
                    gt[classname][img_id] = []
                gt[classname][img_id].append(bbox)

        rec = {}
        prec = {}
        ap = {}
        p = Pool(processes=10)
        ret_values = p.map(eval_det_cls_wrapper,
                        [(pred[classname], gt[classname], ovthresh, use_07_metric, get_iou_func) for classname in
                            gt.keys() if classname in pred])
        p.close()
        for i, classname in enumerate(gt.keys()):
            if classname in pred:
                rec[classname], prec[classname], ap[classname] = ret_values[i]
            else:
                rec[classname] = 0
                prec[classname] = 0
                ap[classname] = 0
            # print(classname, ap[classname])

        return rec, prec, ap


    def eval_grounding(self, pred_all, gt_all, ovthresh=0.25):
        """ Generic functions to compute accuracy for grounding
            Input:
                pred_all: map of {img_id: [(classname, bbox, score)]}
                gt_all: map of {img_id: [(classname, bbox)]}
                ovthresh: scalar, iou threshold
                use_07_metric: bool, if true use VOC07 11 point method
            Output:
                rec: {classname: rec}
                acc: accuracy
        """
        k = (1, 5, 10)
        # k = ('exact', 3, 5)
        dataset2score = {k_: 0.0 for k_ in k}
        dataset2count = 0.0
        for img_id in pred_all.keys():
            target = gt_all[img_id]
            prediction = pred_all[img_id]
            assert prediction is not None
            # sort by scores
            sorted_scores_boxes = sorted(
                prediction, key = lambda x: x[2], reverse=True
            )
            _, sorted_boxes, sorted_scores = zip(*sorted_scores_boxes)
            sorted_boxes = torch.cat([self.corners_to_ends(torch.as_tensor(x)).view(1, 6) for x in sorted_boxes])
            target_box = torch.cat([
                self.corners_to_ends(torch.as_tensor(t[1])).view(-1, 6)
                for t in target[:1]
            ])
            giou = self.generalized_box_iou3d(sorted_boxes, target_box)
            for g in range(giou.shape[1]):
                for k_ in k:
                    if k_ == 'exact':
                        if max(giou[:1, g]) >= ovthresh:
                            dataset2score[k_] += 1.0 / giou.shape[1]
                    else:
                        if max(giou[:k_, g]) >= ovthresh:
                            dataset2score[k_] += 1.0 / giou.shape[1]
            dataset2count += 1.0

        for k_ in k:
            try:
                dataset2score[k_] /= dataset2count
            except:
                pass

        # results = sorted([v for k, v in dataset2score.items()])
        # print(f"Accuracy @ 1, 5, 10: {results} @IOU: {ovthresh} \n")

        return dataset2score

    def compute_metrics(self):
        """ Use accumulated predictions and groundtruths to compute Average Precision.
        """
        # st()
        rec, prec, ap = self.eval_det_multiprocessing(self.pred_map_cls, self.gt_map_cls, ovthresh=self.ap_iou_thresh,
                                                 get_iou_func=get_iou_obb)
        ret_dict = {}
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            ret_dict['%s Average Precision' % (clsname)] = ap[key]
        ret_dict['mAP'] = np.mean(list(ap.values()))
        rec_list = []
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            try:
                ret_dict['%s Recall' % (clsname)] = rec[key][-1]
                rec_list.append(rec[key][-1])
            except:
                ret_dict['%s Recall' % (clsname)] = 0
                rec_list.append(0)
        ret_dict['AR'] = np.mean(rec_list)
        return ret_dict

    def compute_accuracy(self):
        """ Calculate accuracy metric for grounding."""
        results = self.eval_grounding(self.pred_map_cls, self.gt_map_cls, ovthresh=self.ap_iou_thresh)
        return results

    def reset(self):
        self.gt_map_cls = {}  # {scan_id: [(classname, bbox)]}
        self.pred_map_cls = {}  # {scan_id: [(classname, bbox, score)]}
        self.scan_cnt = 0


@HOOKS.register_module()
class DetEvaluator(HookBase):

    def __init__(self, 
                 losses=['boxes', 'labels', 'contrastive_align', 'masks']):
        super().__init__()
        self.losses = losses
        self.ap_iou_thresholds = [0.25, 0.5]

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            if (self.trainer.epoch + 1) % self.trainer.cfg.eval_freq == 0:
                self.eval()
            else:
                self.trainer.comm_info["current_metric_value"] = 0.0  # save for saver
                self.trainer.comm_info["current_metric_name"] = "Det_AR50"  # save for saver

    def eval(self):

        self.trainer.model.eval()
        self.trainer.logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

        matcher = HungarianMatcher(1, 0, 2, True)
        set_criterion = SetCriterion(
                matcher=matcher,
                losses=self.losses, eos_coef=0.1, temperature=0.07)
        criterion = compute_hungarian_loss
        dataset_config = ScannetDatasetConfig(18)
        # Used for AP calculation
        CONFIG_DICT = {
            'remove_empty_box': False, 'use_3d_nms': True,
            'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': True,
            'per_class_proposal': True, 'conf_thresh': 0.0,
            'dataset_config': dataset_config,
            'hungarian_loss': True
        }
        stat_dict = {}

        if set_criterion is not None:
            set_criterion.eval()

        prefixes = ['last_', 'proposal_']
        prefixes += [f'{i}head_' for i in range(5)]

        ap_calculator_list = [
            APCalculator(iou_thresh, dataset_config.class2type)
            for iou_thresh in self.ap_iou_thresholds
        ]
        mAPs = [
            [iou_thresh, {k: 0 for k in prefixes}]
            for iou_thresh in self.ap_iou_thresholds
        ]

        batch_pred_map_cls_dict = {k: [] for k in prefixes}
        batch_gt_map_cls_dict = {k: [] for k in prefixes}

        # Main eval branch
        # NOTE char span and token span.
        wordidx = np.array([
            0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 8, 9, 10, 11,
            12, 13, 13, 14, 15, 16, 16, 17, 17, 18, 18
        ])  # 18+1（not mentioned）
        tokenidx = np.array([
            1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19, 21, 23,
            25, 27, 29, 31, 32, 34, 36, 38, 39, 41, 42, 44, 45
        ])  # 18 token span

        val_loader = tqdm(self.trainer.val_loader, ascii=True)
        for batch_idx, batch_data in enumerate(val_loader):
            # note eval
            stat_dict, end_points = self._main_eval_branch(
                batch_idx, batch_data, val_loader, self.trainer.model, stat_dict,
                criterion, set_criterion, self.trainer.logger
            )

            # step score   contrast
            proj_tokens = end_points['proj_tokens']  # (B, tokens, 64)
            proj_queries = end_points['last_proj_queries']  # (B, Q, 64)
            sem_scores = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2))
            sem_scores_ = sem_scores / 0.07  # (B, Q, tokens)
            sem_scores = torch.zeros(sem_scores_.size(0), sem_scores_.size(1), 256)
            sem_scores = sem_scores.to(sem_scores_.device)
            sem_scores[:, :sem_scores_.size(1), :sem_scores_.size(2)] = sem_scores_
            end_points['last_sem_cls_scores'] = sem_scores  # ([B, 256, 256])

            # step
            sem_cls = torch.zeros_like(end_points['last_sem_cls_scores'])[..., :19] # ([B, 256, 19])
            for w, t in zip(wordidx, tokenidx):
                sem_cls[..., w] += end_points['last_sem_cls_scores'][..., t]
            end_points['last_sem_cls_scores'] = sem_cls     # ([B, 256, 19])

            # step Parse predictions
            # for prefix in prefixes:
            prefix = 'last_'
            # pred
            batch_pred_map_cls, _ = parse_predictions(
                end_points, CONFIG_DICT, prefix,
                size_cls_agnostic=True)
            batch_gt_map_cls = self.parse_groundtruths(
                end_points, CONFIG_DICT,
                size_cls_agnostic=True)
            batch_pred_map_cls_dict[prefix].append(batch_pred_map_cls)
            batch_gt_map_cls_dict[prefix].append(batch_gt_map_cls)

        mAP = 0.0
        # for prefix in prefixes:
        prefix = 'last_'
        for (batch_pred_map_cls, batch_gt_map_cls) in zip(
                batch_pred_map_cls_dict[prefix],
                batch_gt_map_cls_dict[prefix]):
            for ap_calculator in ap_calculator_list:
                ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
        
        # Evaluate average precision
        for i, ap_calculator in enumerate(ap_calculator_list):
            metrics_dict = ap_calculator.compute_metrics()
            self.trainer.logger.info(
                '=====================>'
                f'{prefix} IOU THRESH: {self.ap_iou_thresholds[i]}'
                '<====================='
            )
            for key in metrics_dict:
                self.trainer.logger.info(f'{key} {metrics_dict[key]}')
                if key == "AR" and self.ap_iou_thresholds[i] == 0.5:
                    self.trainer.comm_info["current_metric_value"] = metrics_dict[key]  # save for saver
            if prefix == 'last_' and ap_calculator.ap_iou_thresh > 0.3:
                mAP = metrics_dict['mAP']
            mAPs[i][1][prefix] = metrics_dict['mAP']
            ap_calculator.reset()

        for mAP in mAPs:
            self.trainer.logger.info(
                f'IoU[{mAP[0]}]:\t'
                + ''.join([
                    f'{key}: {mAP[1][key]:.4f} \t'
                    for key in sorted(mAP[1].keys())
                ])
            )
        
        self.trainer.comm_info["current_metric_name"] = "Det_AR50"  # save for saver

        self.trainer.logger.info('<<<<<<<<<<<<<<<<< End Testing <<<<<<<<<<<<<<<<<')

    @torch.no_grad()
    def _main_eval_branch(self, batch_idx, batch_data, test_loader, model,
                          stat_dict,
                          criterion, set_criterion, logger):
        # Move to GPU
        inputs = self._to_gpu(batch_data)

        if "train" not in inputs:
            inputs.update({"train": False})
        else:
            inputs["train"] = False

        # STEP Forward pass
        end_points = model(inputs)

        # STEP Compute loss
        _, end_points = self._compute_loss(
            end_points, criterion, set_criterion
        )

        for key in end_points:
            if 'pred_size' in key:
                end_points[key] = torch.clamp(end_points[key], min=1e-6)

        # Accumulate statistics and print out
        stat_dict = self._accumulate_stats(stat_dict, end_points)
        if (batch_idx + 1) % 500 == 0:
            logger.info(f'Eval: [{batch_idx + 1}/{len(test_loader)}]  ')
            logger.info(''.join([
                f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
                for key in sorted(stat_dict.keys())
                if 'loss' in key and 'proposal_' not in key
                and 'last_' not in key and 'head_' not in key
            ]))
        return stat_dict, end_points

    def _accumulate_stats(self, stat_dict, end_points):
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                if isinstance(end_points[key], (float, int)):
                    stat_dict[key] += end_points[key]
                else:
                    stat_dict[key] += end_points[key].item()
        return stat_dict
    
    def _to_gpu(self, data_dict):
        if torch.cuda.is_available():
            for key in data_dict:
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda(non_blocking=True)
        return data_dict

    @staticmethod
    def _compute_loss(end_points, criterion, set_criterion):
        loss, end_points = criterion(
            end_points, 6,
            set_criterion,
            query_points_obj_topk=5
        )
        return loss, end_points

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)

    def parse_groundtruths(self, end_points, config_dict, size_cls_agnostic):
        """
        Parse groundtruth labels to OBB parameters.

        Args:
            end_points: dict
                {center_label, heading_class_label, heading_residual_label,
                size_class_label, size_residual_label, sem_cls_label,
                box_label_mask}
            config_dict: dict
                {dataset_config}

        Returns:
            batch_gt_map_cls: a list  of len == batch_size (BS)
                [gt_list_i], i = 0, 1, ..., BS-1
                where gt_list_i = [(gt_sem_cls, gt_box_params)_j]
                where j = 0, ..., num of objects - 1 at sample input i
                [
                    [(gt_sem_cls, gt_box_params)_j for j in range(n_obj[i])]
                    for i in range(B)
                ]
        """
        center_label = end_points['center_label']
        if size_cls_agnostic:
            size_gts = end_points['size_gts']
        else:
            size_class_label = end_points['size_class_label']
            size_residual_label = end_points['size_residual_label']
        box_label_mask = end_points['box_label_mask']
        sem_cls_label = end_points['sem_cls_label']
        bsize = center_label.shape[0]

        K2 = center_label.shape[1]  # K2==MAX_NUM_OBJ
        gt_corners_3d_upright_camera = np.zeros((bsize, K2, 8, 3))
        gt_center_upright_camera = flip_axis_to_camera(center_label[:, :, 0:3].detach().cpu().numpy())
        for i in range(bsize):
            for j in range(K2):
                if box_label_mask[i, j] == 0:
                    continue
                heading_angle = 0
                if size_cls_agnostic:
                    box_size = size_gts[i, j].detach().cpu().numpy()
                else:
                    box_size = config_dict['dataset_config'].class2size(int(size_class_label[i, j].detach().cpu().numpy()),
                                                                        size_residual_label[i, j].detach().cpu().numpy())
                corners_3d_upright_camera = get_3d_box(box_size, heading_angle, gt_center_upright_camera[i, j, :])
                gt_corners_3d_upright_camera[i, j] = corners_3d_upright_camera

        batch_gt_map_cls = []
        for i in range(bsize):
            batch_gt_map_cls.append([
                (sem_cls_label[i, j].item(), gt_corners_3d_upright_camera[i, j])
                for j in range(gt_corners_3d_upright_camera.shape[1])
                if box_label_mask[i, j] == 1
            ])
        end_points['batch_gt_map_cls'] = batch_gt_map_cls

        return batch_gt_map_cls

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("allAcc", self.trainer.best_metric_value)
        )
