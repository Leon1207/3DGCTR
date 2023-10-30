# ------------------------------------------------------------------------
# Modification: EDA
# Created: 05/21/2022
# Author: Yanmin Wu
# E-mail: wuyanminmax@gmail.com
# https://github.com/yanmin-wu/EDA 
# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
"""Dataset and data loader."""

import csv
from collections import defaultdict
import h5py
import json
import multiprocessing as mp
import os
import random
from six.moves import cPickle

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast
from .builder import DATASETS
from .transform import Compose, TRANSFORMS

import copy

from .preprocessing.scanrefer.model_util_scannet_v2c import ScannetDatasetConfig_V2C
from .preprocessing.scanrefer.scannet_utils import read_label_mapping
from .preprocessing.scanrefer.visual_data_handlers import Scan, S3D, S3DView
from .preprocessing.scanrefer.scannet_classes import REL_ALIASES, VIEW_DEP_RELS
import wandb
from typing import List, Dict
import tqdm

# NOTE sng_parser
import sys, os
sys.path.append(os.getcwd())
from .preprocessing.scanrefer import sng_parser

NUM_CLASSES = 485
DC = ScannetDatasetConfig_V2C(NUM_CLASSES)
DC18 = ScannetDatasetConfig_V2C(18)
MAX_NUM_OBJ = 132


@DATASETS.register_module()
class Joint3DDataset_Pretrain(Dataset):
    """Dataset utilities for ReferIt3D."""

    def __init__(self, 
                 split='train',
                 data_root='./',
                 transform=None,
                #  dataset_dict={'scanrefer': 1, 'scannet': 10},  # training on rec
                #  test_dataset='scanrefer',
                 dataset_dict={'structured3d': 1},  # pretrain with s3d 
                 test_dataset='structured3d',
                 overfit=False,
                 use_color=True, use_height=False, use_multiview=False,
                 detect_intermediate=True,
                 butd=False, butd_gt=False, butd_cls=False, augment_det=True,
                 wo_obj_name="None", test_mode=False, test_cfg=None, loop=1):
        """Initialize dataset (here for ReferIt3D utterances)."""
        self.dataset_dict = dataset_dict
        self.test_dataset = test_dataset
        self.split = split
        self.use_color = use_color
        self.use_height = use_height
        self.overfit = overfit
        self.detect_intermediate = detect_intermediate
        self.augment = self.split == 'train'
        self.test_mode = test_mode
        self.test_cfg = test_cfg
        self.transform = Compose(transform)
        self.use_multiview = use_multiview
        self.data_path = data_root
        self.butd = False
        self.butd_gt = False
        self.butd_cls = False
        self.loop = loop if not test_mode else 1
        self.joint_det = (  # joint usage of detection/grounding phrases
            'scannet' in dataset_dict
            and len(dataset_dict.keys()) > 1
            and self.split == 'train'
        )
        self.joint_det = (  # only s3d pretrain
            'structured3d' in dataset_dict
            and self.split == 'train'
        )
        self.augment_det = augment_det
        self.wo_obj_name = wo_obj_name
        self.vis_s3d = False

        self.mean_rgb = np.array([109.8, 97.2, 83.8]) / 256
        
        # step 1. semantic label
        self.label_map = read_label_mapping(
            'pointcept/datasets/preprocessing/scanrefer/meta_data/scannetv2-labels.combined.tsv',
            label_from='raw_category',
            label_to='id'
        )
        self.label_map18 = read_label_mapping(
            'pointcept/datasets/preprocessing/scanrefer/meta_data/scannetv2-labels.combined.tsv',
            label_from='raw_category',
            label_to='nyu40id'
        )
        self.label_mapclass = read_label_mapping(
            'pointcept/datasets/preprocessing/scanrefer/meta_data/scannetv2-labels.combined.tsv',
            label_from='raw_category',
            label_to='nyu40class'
        )   

        self.multiview_path = os.path.join(
            f'{self.data_path}/scanrefer_2d_feats',
            "enet_feats_maxpool.hdf5"
        )
        self.multiview_data = {}

        # step 2. transformer tokenizer
        # # 1) online
        # self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        # 2) offline
        self.tokenizer = RobertaTokenizerFast.from_pretrained(f'{self.data_path}/roberta-base/', local_files_only=True)

        if os.path.exists('data/cls_results.json'):
            with open('data/cls_results.json') as fid:
                self.cls_results = json.load(fid)

        print('Loading %s files, take a breath!' % split)
        
        # step 3. generate or load train/val_v3scans.pkl
        self.scans = {}

        # step 4. load datasets for structured3D
        self.annos = []

        # using for processing datasets and save pkl for each scene_room
        s3d_data_path = f'/userhome/lyd/Pointcept/data/structured3d/Only_view/{split}'
        total = 3000 if split == 'train' else 250
        for cnt, (dirpath, _, filenames) in enumerate(os.walk(s3d_data_path)):
            print("Process: {}/{}".format(cnt, total))
            for filename in filenames:
                scene_id = dirpath.split('/')[-1].split('_')[-1] + '_' + filename.split('_')[1] + '_' + filename.split('_')[2].split('.')[0]
                scan = S3DView(scene_id, s3d_data_path)

                keep = np.where(np.array([
                    self.label_map18[
                        scan.get_object_instance_label(ind)
                    ] in DC18.nyu40id2class
                    for ind in range(len(scan.three_d_objects))
                ])[:MAX_NUM_OBJ])[0].tolist()

                if len(keep) > 0:
                    self.annos.append({
                        'scan_id': scene_id,
                        'target_id': [],
                        'distractor_ids': [],
                        'utterance': '',
                        'target': [],
                        'anchors': [],
                        'anchor_ids': [],
                        'dataset': 'structured3d'
                    })
                    pkl_name = scene_id + ".pkl"
                    save_dict = {scene_id: scan}
                    pickle_data(f'/userhome/lyd/Pointcept/data/structured3d/Only_view/{split}_scene_pkl/' + pkl_name, save_dict)

                else:
                    print("Fliter scene: ", scene_id.split(".")[0])

        print("Finished.")
        
        # using for myself with 10 train set split scene lists
        # s3d_pkl_path = f'/userhome/lyd/Pointcept/data/structured3d/Only_panorama/'
        # for cnt in range(1, 11):
        #     print("Processing... ", cnt)
        #     pkl_name = s3d_pkl_path + "train_s3ds_" + str(cnt) + ".pkl"
        #     scans = unpickle_data(pkl_name)
        #     scans = list(scans)[0]
        #     for key in scans.keys():
        #         scan = scans[key]
        #         save_dict = {key: scan}
        #         pickle_data(f'/userhome/lyd/Pointcept/data/structured3d/Only_panorama/train_scene_pkl/' + key + ".pkl", save_dict)
        
        # using for myself with 1 val set split scene lists
        # s3d_pkl_path = f'/userhome/lyd/Pointcept/data/structured3d/Only_panorama/'
        # pkl_name = s3d_pkl_path + "val_s3ds.pkl"
        # scans = unpickle_data(pkl_name)
        # scans = list(scans)[0]
        # for key in scans.keys():
        #     scan = scans[key]
        #     save_dict = {key: scan}
        #     pickle_data(f'/userhome/lyd/Pointcept/data/structured3d/Only_panorama/val_scene_pkl/' + key + ".pkl", save_dict)

        # using after processing, loading each annos and filter invalid scene
        # use_pano_psrp = "Only_view"
        # lis_dir = os.listdir(f'/userhome/lyd/Pointcept/data/structured3d/{use_pano_psrp}/{split}_scene_pkl/')
        # for cnt, scene_id in enumerate(lis_dir):
        #     if cnt % 1000 == 0:
        #         print("Filter process: {}/{}".format(cnt, len(lis_dir)))
        #     scan = unpickle_data(f'/userhome/lyd/Pointcept/data/structured3d/{use_pano_psrp}/{split}_scene_pkl/' + scene_id)
        #     scan = list(scan)[0][scene_id.split(".")[0]]

        #     keep = np.where(np.array([
        #             self.label_map18[
        #                 scan.get_object_instance_label(ind)
        #             ] in DC18.nyu40id2class
        #             for ind in range(len(scan.three_d_objects))
        #         ])[:MAX_NUM_OBJ])[0].tolist()

        #     if len(keep) > 0:
        #         self.annos.append({
        #             'scan_id': scene_id.split(".")[0],
        #             'target_id': [],
        #             'distractor_ids': [],
        #             'utterance': '',
        #             'target': [],
        #             'anchors': [],
        #             'anchor_ids': [],
        #             'dataset': 'structured3d'
        #         })
        #     else:
        #         print("Fliter scene: ", scene_id.split(".")[0])

            # self.annos = self.annos[:100]  # debug
            
    
    # BRIEF smaple classes for detection prompt
    def _sample_classes(self, scan_id):
        """Sample classes for the scannet detection sentences."""
        scan = self.scans[scan_id]
        sampled_classes = set([
            self.label_map[scan.get_object_instance_label(ind)]  # chair, bed, ...
            for ind in range(len(scan.three_d_objects))
        ])
        sampled_classes = list(sampled_classes & set(DC.nyu40id2class))
        # sample 10 classes
        if self.split == 'train' and self.random_utt:  # random utterance
            if len(sampled_classes) > 10:
                sampled_classes = random.sample(sampled_classes, 10)
            ret = [DC.class2type[DC.nyu40id2class[i]] for i in sampled_classes]
            random.shuffle(ret)
        else:
            # ret = [
            #     'cabinet', 'bed', 'chair', 'couch', 'table', 'door',
            #     'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
            #     'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub',
            #     'other furniture'
            # ]
            ret = [
                'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
                'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub',
                'others'
            ]
        return ret

    def _sample_classes_s3d(self, scan):
        """Sample classes for the scannet detection sentences."""
        sampled_classes = set([
            self.label_map[scan.get_object_instance_label(ind)]  # chair, bed, ...
            for ind in range(len(scan.three_d_objects))
        ])
        sampled_classes = list(sampled_classes & set(DC.nyu40id2class))
        # sample 10 classes
        if self.split == 'train' and self.random_utt:  # random utterance
            if len(sampled_classes) > 10:
                sampled_classes = random.sample(sampled_classes, 10)
            ret = [DC.class2type[DC.nyu40id2class[i]] for i in sampled_classes]
            random.shuffle(ret)
        else:
            # ret = [
            #     'cabinet', 'bed', 'chair', 'couch', 'table', 'door',
            #     'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
            #     'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub',
            #     'other furniture'
            # ]
            ret = [
                'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
                'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub',
                'others'
            ]
        return ret
    
    # BRIEF constract utterance for scannet
    def _create_scannet_utterance(self, sampled_classes):
        if self.split == 'train' and self.random_utt:
            neg_names = []
            while len(neg_names) < 10:
                _ind = np.random.randint(0, len(DC.class2type))
                if DC.class2type[_ind] not in neg_names + sampled_classes:
                    neg_names.append(DC.class2type[_ind])
            mixed_names = sorted(list(set(sampled_classes + neg_names)))
            random.shuffle(mixed_names)
        else:
            mixed_names = sampled_classes
        utterance = ' . '.join(mixed_names)
        return utterance

    def _load_multiview(self, scan_id):
        """Load multi-view data of given scan-id."""
        pid = mp.current_process().pid
        if pid not in self.multiview_data:
            self.multiview_data[pid] = h5py.File(
                self.multiview_path, "r", libver="latest"
            )
        return self.multiview_data[pid][scan_id]
    
    # BRIEF point cloud augmentation
    def _augment(self, pc, color, rotate):
        augmentations = {}

        # Rotate/flip only if we don't have a view_dep sentence
        if rotate:
            theta_z = 90*np.random.randint(0, 4) + (2*np.random.rand() - 1) * 5
            # Flipping along the YZ plane
            augmentations['yz_flip'] = np.random.random() > 0.5
            if augmentations['yz_flip']:
                pc[:, 0] = -pc[:, 0]
            # Flipping along the XZ plane
            augmentations['xz_flip'] = np.random.random() > 0.5
            if augmentations['xz_flip']:
                pc[:, 1] = -pc[:, 1]
        else:
            theta_z = (2*np.random.rand() - 1) * 5
        augmentations['theta_z'] = theta_z
        pc[:, :3] = rot_z(pc[:, :3], theta_z)
        # Rotate around x
        theta_x = (2*np.random.rand() - 1) * 2.5
        augmentations['theta_x'] = theta_x
        pc[:, :3] = rot_x(pc[:, :3], theta_x)
        # Rotate around y
        theta_y = (2*np.random.rand() - 1) * 2.5
        augmentations['theta_y'] = theta_y
        pc[:, :3] = rot_y(pc[:, :3], theta_y)

        # Add noise
        noise = np.random.rand(len(pc), 3) * 5e-3
        augmentations['noise'] = noise
        pc[:, :3] = pc[:, :3] + noise

        # Translate/shift
        augmentations['shift'] = np.random.random((3,))[None, :] - 0.5
        pc[:, :3] += augmentations['shift']

        # Scale
        augmentations['scale'] = 0.98 + 0.04*np.random.random()
        pc[:, :3] *= augmentations['scale']

        # Color
        if color is not None:
            color += self.mean_rgb
            color *= 0.98 + 0.04*np.random.random((len(color), 3))
            color -= self.mean_rgb
        return pc, color, augmentations
    
    # BRIEF get point clouds
    def _get_pc(self, anno, scan):
        """Return a point cloud representation of current scene."""
        scan_id = anno['scan_id']
        rel_name = "none"
        if anno['dataset'].startswith('sr3d'):
            rel_name = self._find_rel(anno['utterance'])

        # a. Color
        color = None
        if self.use_color:
            color = scan.color - self.mean_rgb

        # b .Height
        height = None
        if self.use_height:
            floor_height = np.percentile(scan.pc[:, 2], 0.99)
            height = np.expand_dims(scan.pc[:, 2] - floor_height, 1)

        # c. Multi-view 2d features
        multiview_data = None
        if self.use_multiview:
            multiview_data = self._load_multiview(scan_id)

        # d. Augmentations
        augmentations = {}
        if self.split == 'train' and self.augment:
            rotate_natural = (
                anno['dataset'] in ('nr3d', 'scanrefer')
                and self._augment_nr3d(anno['utterance'])
            )
            rotate_sr3d = (
                anno['dataset'].startswith('sr3d')
                and rel_name not in VIEW_DEP_RELS
            )
            rotate_else = (anno['dataset'] == 'scannet') or (anno['dataset'] == 'structured3d')
            rotate = rotate_sr3d or rotate_natural or rotate_else
            pc, color, augmentations = self._augment(scan.pc, color, rotate)
            scan.pc = pc

        # e. Concatenate representations
        point_cloud = scan.pc
        if color is not None:
            point_cloud = np.concatenate((point_cloud, color), 1)
        if height is not None:
            point_cloud = np.concatenate([point_cloud, height], 1)
        if multiview_data is not None:
            point_cloud = np.concatenate([point_cloud, multiview_data], 1)

        return point_cloud, augmentations, scan.color
    
    # BRIEF get position label [scannet]
    def _get_token_positive_map(self, anno):
        """Return correspondence of boxes to tokens."""
        # Token start-end span in characters
        caption = ' '.join(anno['utterance'].replace(',', ' ,').split())
        caption = ' ' + caption + ' '
        
        tokens_positive = np.zeros((MAX_NUM_OBJ, 2))
        if isinstance(anno['target'], list):
            cat_names = anno['target']
        else:
            cat_names = [anno['target']]
        if self.detect_intermediate:
            cat_names += anno['anchors']
        for c, cat_name in enumerate(cat_names):
            start_span = caption.find(' ' + cat_name + ' ')
            len_ = len(cat_name)
            if start_span < 0:
                start_span = caption.find(' ' + cat_name)
                len_ = len(caption[start_span+1:].split()[0])
            if start_span < 0:
                start_span = caption.find(cat_name)
                orig_start_span = start_span
                while caption[start_span - 1] != ' ':
                    start_span -= 1
                len_ = len(cat_name) + orig_start_span - start_span
                while caption[len_ + start_span] != ' ':
                    len_ += 1
            
            end_span = start_span + len_
            assert start_span > -1, caption
            assert end_span > 0, caption
            tokens_positive[c][0] = start_span
            tokens_positive[c][1] = end_span

        # Positive map (for soft token prediction)
        tokenized = self.tokenizer.batch_encode_plus(
            [' '.join(anno['utterance'].replace(',', ' ,').split())],
            padding="longest", return_tensors="pt"
        )

        positive_map = np.zeros((MAX_NUM_OBJ, 256))

        # note: empty in scannet prompt
        modify_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        pron_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        other_entity_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        rel_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        auxi_entity_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        
        # main object component
        gt_map = get_positive_map(tokenized, tokens_positive[:len(cat_names)])
        positive_map[:len(cat_names)] = gt_map
        return tokens_positive, positive_map, modify_positive_map, pron_positive_map, \
            other_entity_positive_map, auxi_entity_positive_map, rel_positive_map


    #################################################
    # BRIEF Get text position label by text parsing #
    #################################################
    def _get_token_positive_map_by_parse(self, anno, auxi_box):
        caption = ' '.join(anno['utterance'].replace(',', ' ,').split())

        # node and edge
        graph_node = anno["graph_node"]
        graph_edge = anno["graph_edge"]

        # step main/modify(attri)/pron/other(auxi)/rel
        target_char_span = np.zeros((MAX_NUM_OBJ, 2))   # target(main)
        modify_char_span = np.zeros((MAX_NUM_OBJ, 2))   # modify(attri)
        pron_char_span = np.zeros((MAX_NUM_OBJ, 2))     # pron
        rel_char_span = np.zeros((MAX_NUM_OBJ, 2))      # rel
        assert graph_node[0]['node_id'] == 0
        main_entity_target = graph_node[0]['target_char_span']
        main_entity_modify = graph_node[0]['mod_char_span']
        main_entity_pron   = graph_node[0]['pron_char_span']
        main_entity_rel   = graph_node[0]['rel_char_span']

        # other(auxi) object token
        other_target_char_span = np.zeros((MAX_NUM_OBJ, 2))
        other_entity_target = []
        if len(graph_node) > 1:
            for node in graph_node:
                if node["node_id"] != 0 and node["node_type"] == "Object":
                    for span in node['target_char_span']:
                        other_entity_target.append(span)

        num_t = 0
        num_m = 0
        num_p = 0
        num_o = 0
        num_r = 0
        # target(main obj.) token
        for t, target in enumerate(main_entity_target):
            target_char_span[t] = target
            num_t = t+1
        # modify(attribute) token
        for m, modify in enumerate(main_entity_modify):
            modify_char_span[m] = modify
            num_m = m+1
        # pron token
        for p, pron in enumerate(main_entity_pron):
            pron_char_span[p] = pron
            num_p = p+1
        for o, other in enumerate(other_entity_target):
            other_target_char_span[o] = other
            num_o = o+1
        # rel token add 0727
        for r, rel in enumerate(main_entity_rel):
            rel_char_span[r] = rel
            num_r = r+1

        tokenized = self.tokenizer.batch_encode_plus(
            [' '.join(anno['utterance'].replace(',', ' ,').split())],
            padding="longest", return_tensors="pt"
        )

        target_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        modify_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        pron_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        other_entity_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        rel_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        gt_map_t = get_positive_map(tokenized, target_char_span[:num_t])
        gt_map_m = get_positive_map(tokenized, modify_char_span[:num_m])
        gt_map_p = get_positive_map(tokenized, pron_char_span[:num_p])
        gt_map_o = get_positive_map(tokenized, other_target_char_span[:num_o])
        gt_map_r = get_positive_map(tokenized, rel_char_span[:num_r])
        
        gt_map_t = gt_map_t.sum(axis=0)
        gt_map_m = gt_map_m.sum(axis=0)
        gt_map_p = gt_map_p.sum(axis=0)
        gt_map_o = gt_map_o.sum(axis=0)
        gt_map_r = gt_map_r.sum(axis=0)

        # NOTE text position label
        target_positive_map[:1] = gt_map_t          # main object
        modify_positive_map[:1] = gt_map_m          # attribute
        pron_positive_map[:1]   = gt_map_p          # pron
        other_entity_positive_map[:1] = gt_map_o    # auxi obj
        rel_positive_map[:1]   = gt_map_r           # relation

        # auxi
        auxi_entity_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        if auxi_box is not None:
            auxi_entity = anno["auxi_entity"]['target_char_span']
            num_a = 0
            # char span
            auxi_entity_char_span = np.zeros((MAX_NUM_OBJ, 2))
            for a, auxi in enumerate(auxi_entity):
                auxi_entity_char_span[a] = auxi
                num_a = a+1
            # position label
            gt_map_a = get_positive_map(tokenized, auxi_entity_char_span[:num_a])
            gt_map_a = gt_map_a.sum(axis=0)
            auxi_entity_positive_map[:1] = gt_map_a

            # note SR3D 
            if anno['dataset'] == 'sr3d':
                target_positive_map[1] = gt_map_a

        return target_char_span, target_positive_map, modify_positive_map, pron_positive_map, \
            other_entity_positive_map, auxi_entity_positive_map, rel_positive_map


    # BRIEF get GT Box.
    def _get_target_boxes(self, anno, scan):
        """Return gt boxes to detect."""
        bboxes = np.zeros((MAX_NUM_OBJ, 6))
        if isinstance(anno['target_id'], list):
            tids = anno['target_id']
        else:  # referit dataset
            tids = [anno['target_id']]
            # TODO SR3D: anchor object
            if self.detect_intermediate:
                # tids += anno.get('anchor_ids', [])    # BUTD-DETR
                # EDA
                if anno['auxi_entity'] is not None and len(anno['anchor_ids']):
                    tids.append(anno['anchor_ids'][0])
        point_instance_label = -np.ones(len(scan.pc))
        gt_masks = np.zeros((MAX_NUM_OBJ, len(scan.pc)))
        for t, tid in enumerate(tids):
            point_instance_label[scan.three_d_objects[tid]['points']] = t
            gt_masks[t][scan.three_d_objects[tid]['points']] = 1
        
        bboxes[:len(tids)] = np.stack([
            scan.get_object_bbox(tid).reshape(-1) for tid in tids
        ])
        bboxes = np.concatenate((
            (bboxes[:, :3] + bboxes[:, 3:]) * 0.5,
            bboxes[:, 3:] - bboxes[:, :3]
        ), 1)
        if self.split == 'train' and self.augment and anno['dataset'] != 'structured3d':  # jitter boxes
            bboxes[:len(tids)] *= (0.95 + 0.1*np.random.random((len(tids), 6)))
        bboxes[len(tids):, :3] = 1000
        
        box_label_mask = np.zeros(MAX_NUM_OBJ)
        box_label_mask[:len(tids)] = 1
        
        return bboxes, box_label_mask, point_instance_label, gt_masks

    def _get_scene_objects(self, scan):
        # Objects to keep
        keep_ = np.array([
            self.label_map[
                scan.get_object_instance_label(ind)
            ] in DC.nyu40id2class
            for ind in range(len(scan.three_d_objects))
        ])[:MAX_NUM_OBJ]    # keep_ (object_num)
        keep = np.array([False] * MAX_NUM_OBJ)
        keep[:len(keep_)] = keep_

        # Class ids 
        cid = np.array([
            DC.nyu40id2class[self.label_map[scan.get_object_instance_label(k)]]
            for k, kept in enumerate(keep) if kept
        ])
        class_ids = np.zeros((MAX_NUM_OBJ,))
        class_ids[keep] = cid

        # constract object boxes
        all_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        all_bboxes_ = np.stack([
            scan.get_object_bbox(k).reshape(-1)
            for k, kept in enumerate(keep) if kept
        ])
        # cx, cy, cz, w, h, d
        all_bboxes_ = np.concatenate((
            (all_bboxes_[:, :3] + all_bboxes_[:, 3:]) * 0.5,
            all_bboxes_[:, 3:] - all_bboxes_[:, :3]
        ), 1)
        all_bboxes[keep] = all_bboxes_
        if self.split == 'train' and self.augment:
            all_bboxes *= (0.95 + 0.1*np.random.random((len(all_bboxes), 6)))

        # Which boxes we're interested for
        all_bbox_label_mask = keep
        return class_ids, all_bboxes, all_bbox_label_mask
    

    # BRIEF Search for pseudo-labels of auxiliary objects [not used!]
    def _get_auxi_boxes(self, anno, class_ids, all_bboxes, all_bbox_label_mask, gt_bboxes):
        auxi_box = None

        if anno["dataset"] == "scannet":
            return auxi_box

        if anno["auxi_entity"] is not None:
            auxi_label_lemma = anno["auxi_entity"]["lemma_head"]
            if auxi_label_lemma in self.label_map:
                if self.label_map[auxi_label_lemma] not in list((DC.nyu40id2class).keys()):
                    return auxi_box
                
                cls_id = DC.nyu40id2class[self.label_map[auxi_label_lemma]]
                dis_min = 100
                target_box = gt_bboxes[0]
                for idx, mask in enumerate(all_bbox_label_mask):
                    if anno['target_id'] == idx or mask == False:
                        continue
                    if class_ids[idx] == cls_id:
                        dis = target_box[:3] - all_bboxes[idx][:3]
                        dis = np.sum(dis**2)
                        
                        if dis < dis_min:
                            dis_min = dis
                            auxi_box = all_bboxes[idx]
        return auxi_box

    # BRIEF GroupFree detection boxes
    def _get_detected_objects(self, split, scan_id, augmentations):
        # Initialize
        all_detected_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        all_detected_bbox_label_mask = np.array([False] * MAX_NUM_OBJ)
        detected_class_ids = np.zeros((MAX_NUM_OBJ,))
        detected_logits = np.zeros((MAX_NUM_OBJ, NUM_CLASSES))

        # note single stage method
        if not self.butd and not self.butd_cls:
            return (
                all_detected_bboxes, all_detected_bbox_label_mask,
                detected_class_ids, detected_logits
            )

        # Load: class, box, pc, logits
        detected_dict = np.load(
            f'{self.data_path}/group_free_pred_bboxes/group_free_pred_bboxes_{split}/{scan_id}.npy',
            allow_pickle=True
        ).item()

        all_bboxes_ = np.array(detected_dict['box'])
        classes = detected_dict['class']
        cid = np.array([DC.nyu40id2class[
            self.label_map[c]] for c in detected_dict['class']
        ])
        all_bboxes_ = np.concatenate((
            (all_bboxes_[:, :3] + all_bboxes_[:, 3:]) * 0.5,
            all_bboxes_[:, 3:] - all_bboxes_[:, :3]
        ), 1)

        assert len(classes) < MAX_NUM_OBJ
        assert len(classes) == all_bboxes_.shape[0]

        num_objs = len(classes)
        all_detected_bboxes[:num_objs] = all_bboxes_
        all_detected_bbox_label_mask[:num_objs] = np.array([True] * num_objs)
        detected_class_ids[:num_objs] = cid
        detected_logits[:num_objs] = detected_dict['logits']    # logits
        # Match current augmentations
        if self.augment and self.split == 'train':
            all_det_pts = box2points(all_detected_bboxes).reshape(-1, 3)
            all_det_pts = rot_z(all_det_pts, augmentations['theta_z'])
            all_det_pts = rot_x(all_det_pts, augmentations['theta_x'])
            all_det_pts = rot_y(all_det_pts, augmentations['theta_y'])
            if augmentations.get('yz_flip', False):
                all_det_pts[:, 0] = -all_det_pts[:, 0]
            if augmentations.get('xz_flip', False):
                all_det_pts[:, 1] = -all_det_pts[:, 1]
            all_det_pts += augmentations['shift']
            all_det_pts *= augmentations['scale']
            all_detected_bboxes = points2box(all_det_pts.reshape(-1, 8, 3))
        
        if self.augment_det and self.split == 'train':
            min_ = all_detected_bboxes.min(0)
            max_ = all_detected_bboxes.max(0)
            rand_box = (
                (max_ - min_)[None]
                * np.random.random(all_detected_bboxes.shape)
                + min_
            )
            corrupt = np.random.random(len(all_detected_bboxes)) > 0.7
            all_detected_bboxes[corrupt] = rand_box[corrupt]
            detected_class_ids[corrupt] = np.random.randint(
                0, len(DC.nyu40ids), (len(detected_class_ids))
            )[corrupt]
        return (
            all_detected_bboxes, all_detected_bbox_label_mask,
            detected_class_ids, detected_logits
        )

    # BRIEF data
    def __getitem__(self, index):
        """Get current batch for input index."""
        split = self.split

        # train dataset
        language_dataset = self.test_dataset

        # step Read annotation and point clouds
        anno = self.annos[index]
        scan = unpickle_data(f'/userhome/lyd/Pointcept/data/structured3d/Only_view/{split}_scene_pkl/' + anno['scan_id'] + '.pkl')
        scan = list(scan)[0][anno['scan_id']]
        scan.pc = np.copy(scan.orig_pc)
        superpoint = torch.zeros((1))  # avoid bugs

        # step constract anno (used only for [scannet])
        self.random_utt = False
        if (anno['dataset'] == 'scannet') or (anno['dataset'] == 'structured3d'):
            self.random_utt = self.joint_det and np.random.random() > 0.5

            sampled_classes = self._sample_classes_s3d(scan)
            utterance = self._create_scannet_utterance(sampled_classes)
            
            if not self.random_utt:  # detection18 phrase
                anno['target_id'] = np.where(np.array([
                    self.label_map18[
                        scan.get_object_instance_label(ind)
                    ] in DC18.nyu40id2class
                    for ind in range(len(scan.three_d_objects))
                ])[:MAX_NUM_OBJ])[0].tolist()  # target_id是相对于整个scene拥有的object的index
            else:
                anno['target_id'] = np.where(np.array([
                    self.label_map[
                        scan.get_object_instance_label(ind)
                    ] in DC.nyu40id2class
                    and
                    DC.class2type[DC.nyu40id2class[self.label_map[
                        scan.get_object_instance_label(ind)
                    ]]] in sampled_classes
                    for ind in range(len(scan.three_d_objects))
                ])[:MAX_NUM_OBJ])[0].tolist()
            
            # Target names
            if not self.random_utt:
                anno['target'] = [
                    DC18.class2type[DC18.nyu40id2class[self.label_map18[
                        scan.get_object_instance_label(ind)
                    ]]]
                    if self.label_map18[
                        scan.get_object_instance_label(ind)
                    ] != 39
                    # else 'other furniture'
                    else 'others'
                    for ind in anno['target_id']
                ]
            else:
                anno['target'] = [
                    DC.class2type[DC.nyu40id2class[self.label_map[
                        scan.get_object_instance_label(ind)
                    ]]]
                    for ind in anno['target_id']
                ]
            anno['utterance'] = utterance
   
        # step Point cloud representation
        point_cloud, augmentations, og_color = self._get_pc(anno, scan)
        offset = torch.tensor([point_cloud.shape[0]])

        # step "Target" boxes: append anchors if they're to be detected
        gt_bboxes, box_label_mask, point_instance_label, gt_masks = \
            self._get_target_boxes(anno, scan)

        # step Scene gt boxes
        (
            class_ids, all_bboxes, all_bbox_label_mask
        ) = self._get_scene_objects(scan)

        # not used
        # auxi_box = self._get_auxi_boxes(anno, class_ids, all_bboxes, all_bbox_label_mask, gt_bboxes)
        auxi_box = None

        ##########################
        # STEP Get text position #
        ##########################
        if (anno['dataset'] == 'scannet') or (anno['dataset'] == 'structured3d'):
            tokens_positive, positive_map, modify_positive_map, pron_positive_map, \
                other_entity_map, auxi_entity_positive_map, rel_positive_map = self._get_token_positive_map(anno)
        else:
            # note text parsing
            tokens_positive, positive_map, modify_positive_map, pron_positive_map, \
                other_entity_map, auxi_entity_positive_map, rel_positive_map = self._get_token_positive_map_by_parse(anno, auxi_box)
        if auxi_box is None:
            auxi_box = np.zeros((1, 6))
        else:
            auxi_box = np.expand_dims(auxi_box, axis=0)
        
        # step groupfree Detected boxes

        all_detected_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        all_detected_bbox_label_mask = np.array([False] * MAX_NUM_OBJ)
        detected_class_ids = np.zeros((MAX_NUM_OBJ,))
        detected_logits = np.zeros((MAX_NUM_OBJ, NUM_CLASSES))

        if anno['dataset'] != 'structured3d':  # s3d not used groupfree
            (
                all_detected_bboxes, all_detected_bbox_label_mask,
                detected_class_ids, detected_logits
            ) = self._get_detected_objects(split, anno['scan_id'], augmentations)

            # Assume a perfect object detector
            if self.butd_gt:
                all_detected_bboxes = all_bboxes
                all_detected_bbox_label_mask = all_bbox_label_mask
                detected_class_ids = class_ids

            # Assume a perfect object proposal stage
            if self.butd_cls:
                all_detected_bboxes = all_bboxes
                all_detected_bbox_label_mask = all_bbox_label_mask
                detected_class_ids = np.zeros((len(all_bboxes,)))
                classes = np.array(self.cls_results[anno['scan_id']])
                detected_class_ids[all_bbox_label_mask] = classes[classes > -1]

        # Return
        _labels = np.zeros(MAX_NUM_OBJ)
        if not isinstance(anno['target_id'], int) and not self.random_utt:
            _labels[:len(anno['target_id'])] = np.array([
                DC18.nyu40id2class[self.label_map18[
                    scan.get_object_instance_label(ind)
                ]]
                for ind in anno['target_id']
            ])

        # vis structured
        if self.vis_s3d == True:
            wandb.init(project="vis_s3d", name="gt_s3d")
            point_cloud_vis = torch.from_numpy(point_cloud.astype(np.float32))
            og_color_vis = torch.from_numpy(og_color.astype(np.float32))
            point_cloud_vis[:, 3:] = (og_color_vis + torch.tensor([109.8, 97.2, 83.8]) / 256) * 256
            blue = torch.tensor([0.0, 0.0, 255.0])
            white = torch.tensor([255.0, 255.0, 255.0])

            utterances = (
                    ' '.join(anno['utterance'].replace(',', ' ,').split())
                    + ' . not mentioned'
                )
            gt_box = box2points(gt_bboxes[..., :6])

            gt_cloud = point_cloud_vis
            gt_mask_idx = gt_masks.astype(np.int64)[0] == 1
            gt_cloud[gt_mask_idx, 3:] = blue

            area_set = set(point_instance_label.tolist())
            ins_cloud = point_cloud_vis
            ins_cloud[..., 3:] = white
            for area in area_set:
                idx = point_instance_label == area
                random_color = (torch.rand(3) * 256.0).float()
                ins_cloud[idx, 3:] = random_color

            wandb.log({
                    "point_scene": wandb.Object3D({
                        "type": "lidar/beta",
                        "points": ins_cloud,
                        "boxes": np.array(
                            [
                                {
                                    "corners": c.tolist(),
                                    "label": anno['target'][idx],
                                    "color": [0, 255, 0]
                                }
                                for idx, c in enumerate(gt_box[:len(anno['target'])])
                            ]
                        )
                    }),
                    "utterance": wandb.Html(utterances),
                })

        ret_dict = {
            'box_label_mask': box_label_mask.astype(np.float32),
            'center_label': gt_bboxes[:, :3].astype(np.float32),
            'sem_cls_label': _labels.astype(np.int64),
            'size_gts': gt_bboxes[:, 3:].astype(np.float32),
            'gt_masks': gt_masks.astype(np.int64),
        }
        ret_dict.update({
            "scan_ids": anno['scan_id'],
            "point_clouds": torch.from_numpy(point_cloud.astype(np.float32)),
            "og_color": og_color.astype(np.float32),
            "utterances": (
                ' '.join(anno['utterance'].replace(',', ' ,').split())
                + ' . not mentioned'
            ),
            "language_dataset": language_dataset,
            "tokens_positive": tokens_positive.astype(np.int64),
            # NOTE text component position label
            "positive_map": positive_map.astype(np.float32),                # main object
            "modify_positive_map": modify_positive_map.astype(np.float32),  # modift(attribute)
            "pron_positive_map": pron_positive_map.astype(np.float32),      # pron
            "other_entity_map": other_entity_map.astype(np.float32),        # other(auxi) object
            "rel_positive_map": rel_positive_map.astype(np.float32),        # relation
            "auxi_entity_positive_map": auxi_entity_positive_map.astype(np.float32),
            "auxi_box":auxi_box.astype(np.float32),
            "relation": (
                self._find_rel(anno['utterance'])
                if anno['dataset'].startswith('sr3d')
                else "none"
            ),
            "target_name": scan.get_object_instance_label(
                anno['target_id'] if isinstance(anno['target_id'], int)
                else anno['target_id'][0]
            ),
            "target_id": (
                anno['target_id'] if isinstance(anno['target_id'], int)
                else anno['target_id'][0]
            ),
            "point_instance_label": point_instance_label.astype(np.int64),
            "all_bboxes": all_bboxes.astype(np.float32),
            "all_bbox_label_mask": all_bbox_label_mask.astype(np.bool8),
            "all_class_ids": class_ids.astype(np.int64),
            "distractor_ids": np.array(
                anno['distractor_ids']
                + [-1] * (32 - len(anno['distractor_ids']))
            ).astype(int),
            "anchor_ids": np.array(
                anno['anchor_ids']
                + [-1] * (32 - len(anno['anchor_ids']))
            ).astype(int),
            "all_detected_boxes": all_detected_bboxes.astype(np.float32),
            "all_detected_bbox_label_mask": all_detected_bbox_label_mask.astype(np.bool8),
            "all_detected_class_ids": detected_class_ids.astype(np.int64),
            "all_detected_logits": detected_logits.astype(np.float32),
            "is_view_dep": self._is_view_dep(anno['utterance']),
            "is_hard": len(anno['distractor_ids']) > 1,
            "is_unique": len(anno['distractor_ids']) == 0,
            "target_cid": (
                class_ids[anno['target_id']]
                if isinstance(anno['target_id'], int)
                else class_ids[anno['target_id'][0]]
            ),
            "offset": offset,
            "superpoint": superpoint,  # avoid bugs
            "source_xzy": point_cloud[..., 0:3].astype(np.float32)
        })

        del scan  # avoid out of memory

        return ret_dict

    @staticmethod
    def _is_view_dep(utterance):
        """Check whether to augment based on nr3d utterance."""
        rels = [
            'front', 'behind', 'back', 'left', 'right', 'facing',
            'leftmost', 'rightmost', 'looking', 'across'
        ]
        words = set(utterance.split())
        return any(rel in words for rel in rels)
    
    @staticmethod
    def _find_rel(utterance):
        utterance = ' ' + utterance.replace(',', ' ,') + ' '
        relation = "none"
        sorted_rel_list = sorted(REL_ALIASES, key=len, reverse=True)
        for rel in sorted_rel_list:
            if ' ' + rel + ' ' in utterance:
                relation = REL_ALIASES[rel]
                break
        return relation
    
    @staticmethod
    def _augment_nr3d(utterance):
        """Check whether to augment based on nr3d utterance."""
        rels = [
            'front', 'behind', 'back', 'left', 'right', 'facing',
            'leftmost', 'rightmost', 'looking', 'across'
        ]
        augment = True
        for rel in rels:
            if ' ' + rel + ' ' in (utterance + ' '):
                augment = False
        return augment
    
    def __len__(self):
        """Return number of utterances."""
        return len(self.annos)

# BRIEF Construct position label(map)
def get_positive_map(tokenized, tokens_positive):
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)  # ([positive], 256])
    for j, tok_list in enumerate(tokens_positive):
        (beg, end) = tok_list
        beg = int(beg)
        end = int(end)
        beg_pos = tokenized.char_to_token(beg)
        end_pos = tokenized.char_to_token(end - 1)
        if beg_pos is None:
            try:
                beg_pos = tokenized.char_to_token(beg + 1)
                if beg_pos is None:
                    beg_pos = tokenized.char_to_token(beg + 2)
            except:
                beg_pos = None
        if end_pos is None:
            try:
                end_pos = tokenized.char_to_token(end - 2)
                if end_pos is None:
                    end_pos = tokenized.char_to_token(end - 3)
            except:
                end_pos = None
        if beg_pos is None or end_pos is None:
            continue
        positive_map[j, beg_pos:end_pos + 1].fill_(1)

    positive_map = positive_map / (positive_map.sum(-1)[:, None] + 1e-12)
    return positive_map.numpy()


def rot_x(pc, theta):
    """Rotate along x-axis."""
    theta = theta * np.pi / 180
    return np.matmul(
        np.array([
            [1.0, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ]),
        pc.T
    ).T


def rot_y(pc, theta):
    """Rotate along y-axis."""
    theta = theta * np.pi / 180
    return np.matmul(
        np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1.0, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ]),
        pc.T
    ).T


def rot_z(pc, theta):
    """Rotate along z-axis."""
    theta = theta * np.pi / 180
    return np.matmul(
        np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1.0]
        ]),
        pc.T
    ).T

def box2points(box):
    """Convert box center/hwd coordinates to vertices (8x3)."""
    x_min, y_min, z_min = (box[:, :3] - (box[:, 3:] / 2)).transpose(1, 0)
    x_max, y_max, z_max = (box[:, :3] + (box[:, 3:] / 2)).transpose(1, 0)
    return np.stack((
        np.concatenate((x_min[:, None], y_min[:, None], z_min[:, None]), 1),
        np.concatenate((x_min[:, None], y_max[:, None], z_min[:, None]), 1),
        np.concatenate((x_max[:, None], y_min[:, None], z_min[:, None]), 1),
        np.concatenate((x_max[:, None], y_max[:, None], z_min[:, None]), 1),
        np.concatenate((x_min[:, None], y_min[:, None], z_max[:, None]), 1),
        np.concatenate((x_min[:, None], y_max[:, None], z_max[:, None]), 1),
        np.concatenate((x_max[:, None], y_min[:, None], z_max[:, None]), 1),
        np.concatenate((x_max[:, None], y_max[:, None], z_max[:, None]), 1)
    ), axis=1)


def points2box(box):
    """Convert vertices (Nx8x3) to box center/hwd coordinates (Nx6)."""
    return np.concatenate((
        (box.min(1) + box.max(1)) / 2,
        box.max(1) - box.min(1)
    ), axis=1)

# BRIEF load scannet
def scannet_loader(iter_obj):
    """Load the scans in memory, helper function."""
    scan_id, scan_path = iter_obj
    print(scan_id)
    return Scan(scan_id, scan_path, True)

# BRIEF Save all scans to pickle.
def save_data(filename, split, data_path):
    """Save all scans to pickle."""
    import multiprocessing as mp

    # Read all scan files
    scan_path = data_path + 'scans/'
    with open('pointcept/datasets/preprocessing/scanrefer/meta_data/scannetv2_%s.txt' % split) as f:
        scan_ids = [line.rstrip() for line in f]    # train/val scene id list.
    print('{} scans found.'.format(len(scan_ids)))

    # Load data
    n_items = len(scan_ids)
    n_processes = 4  # min(mp.cpu_count(), n_items)
    pool = mp.Pool(n_processes)
    chunks = int(n_items / n_processes)
    all_scans = dict()
    iter_obj = [
        (scan_id, scan_path)
        for scan_id in scan_ids
    ]

    for i, data in enumerate(
        pool.imap(scannet_loader, iter_obj, chunksize=chunks)
    ):
        all_scans[scan_ids[i]] = data
    pool.close()
    pool.join()

    # Save data
    print('pickle time')
    pickle_data(filename, all_scans)


def pickle_data(file_name, *args):
    """Use (c)Pickle to save multiple objects in a single file."""
    out_file = open(file_name, 'wb')
    cPickle.dump(len(args), out_file, protocol=2)
    for item in args:
        cPickle.dump(item, out_file, protocol=2)
    out_file.close()

# BRIEF read from pkl
def unpickle_data(file_name, python2_to_3=False):
    """Restore data previously saved with pickle_data()."""
    in_file = open(file_name, 'rb')
    if python2_to_3:
        size = cPickle.load(in_file, encoding='latin1')
    else:
        size = cPickle.load(in_file)

    for _ in range(size):
        if python2_to_3:
            yield cPickle.load(in_file, encoding='latin1')
        else:
            yield cPickle.load(in_file)
    in_file.close()


#########################
# BRIEF Text decoupling #
#########################
def Scene_graph_parse(annos):
    print('Begin text decoupling......')
    for anno in annos:
        caption = ' '.join(anno['utterance'].replace(',', ' , ').split())

        # some error or typo in ScanRefer.
        caption = ' '.join(caption.replace("'m", "am").split())
        caption = ' '.join(caption.replace("'s", "is").split())
        caption = ' '.join(caption.replace("2-tiered", "2 - tiered").split())
        caption = ' '.join(caption.replace("4-drawers", "4 - drawers").split())
        caption = ' '.join(caption.replace("5-drawer", "5 - drawer").split())
        caption = ' '.join(caption.replace("8-hole", "8 - hole").split())
        caption = ' '.join(caption.replace("7-shaped", "7 - shaped").split())
        caption = ' '.join(caption.replace("2-door", "2 - door").split())
        caption = ' '.join(caption.replace("3-compartment", "3 - compartment").split())
        caption = ' '.join(caption.replace("computer/", "computer /").split())
        caption = ' '.join(caption.replace("3-tier", "3 - tier").split())
        caption = ' '.join(caption.replace("3-seater", "3 - seater").split())
        caption = ' '.join(caption.replace("4-seat", "4 - seat").split())
        caption = ' '.join(caption.replace("theses", "these").split())
        
        # some error or typo in NR3D.
        if anno['dataset'] == 'nr3d':
            caption = ' '.join(caption.replace('.', ' .').split())
            caption = ' '.join(caption.replace(';', ' ; ').split())
            caption = ' '.join(caption.replace('-', ' ').split())
            caption = ' '.join(caption.replace('"', ' ').split())
            caption = ' '.join(caption.replace('?', ' ').split())
            caption = ' '.join(caption.replace("*", " ").split())
            caption = ' '.join(caption.replace(':', ' ').split())
            caption = ' '.join(caption.replace('$', ' ').split())
            caption = ' '.join(caption.replace("#", " ").split())
            caption = ' '.join(caption.replace("/", " / ").split())
            caption = ' '.join(caption.replace("you're", "you are").split())
            caption = ' '.join(caption.replace("isn't", "is not").split())
            caption = ' '.join(caption.replace("thats", "that is").split())
            caption = ' '.join(caption.replace("doesn't", "does not").split())
            caption = ' '.join(caption.replace("doesnt", "does not").split())
            caption = ' '.join(caption.replace("itis", "it is").split())
            caption = ' '.join(caption.replace("left-hand", "left - hand").split())
            caption = ' '.join(caption.replace("[", " [ ").split())
            caption = ' '.join(caption.replace("]", " ] ").split())
            caption = ' '.join(caption.replace("(", " ( ").split())
            caption = ' '.join(caption.replace(")", " ) ").split())
            caption = ' '.join(caption.replace("wheel-chair", "wheel - chair").split())
            caption = ' '.join(caption.replace(";s", "is").split())
            caption = ' '.join(caption.replace("tha=e", "the").split())
            caption = ' '.join(caption.replace("it’s", "it is").split())
            caption = ' '.join(caption.replace("’s", " is").split())
            caption = ' '.join(caption.replace("isnt", "is not").split())
            caption = ' '.join(caption.replace("Don't", "Do not").split())
            caption = ' '.join(caption.replace("arent", "are not").split())
            caption = ' '.join(caption.replace("cant", "can not").split())
            caption = ' '.join(caption.replace("you’re", "you are").split())
            caption = ' '.join(caption.replace('!', ' !').split())
            caption = ' '.join(caption.replace('id the', ' , the').split())
            caption = ' '.join(caption.replace('youre', 'you are').split())

            caption = ' '.join(caption.replace("'", ' ').split())

            if caption[0] == "'":
                caption = caption[1:]
            if caption[-1] == "'":
                caption = caption[:-1]
        
        anno['utterance'] = caption

        # text parsing
        graph_node, graph_edge = sng_parser.parse(caption)

        # NOTE If no node is parsed, add "this is an object ." at the beginning of the sentence
        if (len(graph_node) < 1) or \
            (len(graph_node) > 0 and graph_node[0]["node_id"] != 0):
            caption = "This is an object . " + caption
            anno['utterance'] = caption

            # parse again
            graph_node, graph_edge = sng_parser.parse(caption)

        # node and edge
        anno["graph_node"] = graph_node
        anno["graph_edge"] = graph_edge

        # auxi object
        auxi_entity = None
        for node in graph_node:
            if (node["node_id"] != 0) and (node["node_type"] == "Object"):
                auxi_entity = node
                break
        anno["auxi_entity"] = auxi_entity
    
    print('End text decoupling!')