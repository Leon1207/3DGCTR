# Copyright (c) Facebook, Inc. and its affiliates.

""" 
Modified from https://github.com/facebookresearch/votenet
Dataset for object bounding box regression.
An axis aligned bounding box is parameterized by (cx,cy,cz) and (dx,dy,dz)
where (cx,cy,cz) is the center point of the box, dx is the x-axis length of the box.
"""
import os, json, random, tqdm, h5py, pickle
import numpy as np
import torch
import multiprocessing as mp
import pointcept.datasets.pc_util as pc_util

from pointcept.datasets.box_util import (flip_axis_to_camera_np, flip_axis_to_camera_tensor,
                            get_3d_box_batch_np, get_3d_box_batch_tensor)
from pointcept.datasets.pc_util import scale_points, shift_scale_points
from .builder import DATASETS
from transformers import RobertaTokenizerFast

from typing import List, Dict
from collections import defaultdict
from .preprocessing.scanrefer.model_util_scannet_v2c import ScannetDatasetConfig_V2C
from .preprocessing.scanrefer.scannet_utils import read_label_mapping
from six.moves import cPickle

IGNORE_LABEL = -100
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
MAX_NUM_OBJ = 132
NUM_CLASSES = 485
DC = ScannetDatasetConfig_V2C(NUM_CLASSES)

DATA_ROOT = '/data/pointcloud/data_for_vote2cap/data/'  # modify
SCANREFER = {
    'language': {
        'train': json.load(
            open(os.path.join(DATA_ROOT, "nr3d_train.json"), "r")
        ),
        'val': json.load(
            open(os.path.join(DATA_ROOT, "nr3d_val.json"), "r")
        )
    },
    'scene_list': {
        'train': open(os.path.join(
            DATA_ROOT, 'nr3d_train.txt'
        ), 'r').read().split(),
        'val': open(os.path.join(
            DATA_ROOT, 'nr3d_val.txt'
        ), 'r').read().split()
    },
    'vocabulary': json.load(
        open(os.path.join(DATA_ROOT, "nr3d_vocabulary.json"), "r")
    )
}
vocabulary = SCANREFER['vocabulary']


class ScanReferTokenizer:
    def __init__(self, word2idx: Dict):
        self.word2idx = {word: int(index) for word, index in word2idx.items()}
        self.idx2word = {int(index): word for word, index in word2idx.items()}
        
        self.pad_token = None
        self.bos_token = 'sos'
        self.bos_token_id = word2idx[self.bos_token]
        self.eos_token = 'eos'
        self.eos_token_id = word2idx[self.eos_token]
        
    def __len__(self) -> int: return len(self.word2idx)
    
    def __call__(self, token: str) -> int: 
        token = token if token in self.word2idx else 'unk'
        return self.word2idx[token]
    
    def encode(self, sentence: str) -> List:
        if not sentence: 
            return []
        return [self(word) for word in sentence.split(' ')]
    
    def batch_encode_plus(
        self, sentences: List[str], max_length: int=None, **tokenizer_kwargs: Dict
    ) -> Dict:
        
        raw_encoded = [self.encode(sentence) for sentence in sentences]
        
        if max_length is None:  # infer if not presented
            max_length = max(map(len, raw_encoded))
            
        token = np.zeros((len(raw_encoded), max_length))
        masks = np.zeros((len(raw_encoded), max_length))
        
        for batch_id, encoded in enumerate(raw_encoded):
            length = min(len(encoded), max_length)
            if length > 0:
                token[batch_id, :length] = encoded[:length]
                masks[batch_id, :length] = 1
        
        if tokenizer_kwargs['return_tensors'] == 'pt':
            token, masks = torch.from_numpy(token), torch.from_numpy(masks)
        
        return {'input_ids': token, 'attention_mask': masks}
    
    def decode(self, tokens: List[int]) -> List[str]:
        out_words = []
        for token_id in tokens:
            if token_id == self.eos_token_id: 
                break
            out_words.append(self.idx2word[token_id])
        return ' '.join(out_words)
    
    def batch_decode(self, list_tokens: List[int], **kwargs) -> List[str]:
        return [self.decode(tokens) for tokens in list_tokens]


class DatasetConfig(object):
    def __init__(self):
        self.num_semcls = 18
        self.num_angle_bin = 1
        self.max_num_obj = 132
        self.meta_data_dir = os.path.join(DATA_ROOT, "scannet", "meta_data")

        self.type2class = {
            'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
            'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 
            'curtain':11, 'refrigerator':12, 'shower curtain':13, 'toilet':14, 
            'sink':15, 'bathtub':16, 'others':17
        }
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        self.nyu40ids = np.array([
            3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 
            18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 
            32, 33, 34, 35, 36, 37, 38, 39, 40
        ])
        self.nyu40id2class = {
            nyu40id: i for i, nyu40id in enumerate(list(self.nyu40ids))
        }
        self.nyu40id2class = self._get_nyu40id2class()

    def _get_nyu40id2class(self):
        lines = [line.rstrip() for line in open(os.path.join(self.meta_data_dir, 'scannetv2-labels.combined.tsv'))]
        lines = lines[1:]
        nyu40ids2class = {}
        for i in range(len(lines)):
            label_classes_set = set(self.type2class.keys())
            elements = lines[i].split('\t')
            nyu40_id = int(elements[4])
            nyu40_name = elements[7]
            if nyu40_id in self.nyu40ids:
                if nyu40_name not in label_classes_set:
                    nyu40ids2class[nyu40_id] = self.type2class["others"]
                else:
                    nyu40ids2class[nyu40_id] = self.type2class[nyu40_name]
        return nyu40ids2class


    def angle2class(self, angle):
        raise ValueError("ScanNet does not have rotated bounding boxes.")

    def class2anglebatch_tensor(self, pred_cls, residual, to_label_format=True):
        zero_angle = torch.zeros(
            (pred_cls.shape[0], pred_cls.shape[1]),
            dtype=torch.float32,
            device=pred_cls.device,
        )
        return zero_angle

    def class2anglebatch(self, pred_cls, residual, to_label_format=True):
        zero_angle = np.zeros(pred_cls.shape[0], dtype=np.float32)
        return zero_angle

    def param2obb(
        self,
        center,
        heading_class,
        heading_residual,
        size_class,
        size_residual,
        box_size=None,
    ):
        heading_angle = self.class2angle(heading_class, heading_residual)
        if box_size is None:
            box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle * -1
        return obb

    def box_parametrization_to_corners(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_tensor(box_center_unnorm)
        boxes = get_3d_box_batch_tensor(box_size, box_angle, box_center_upright)
        return boxes

    def box_parametrization_to_corners_np(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_np(box_center_unnorm)
        boxes = get_3d_box_batch_np(box_size, box_angle, box_center_upright)
        return boxes

    @staticmethod
    def rotate_aligned_boxes(input_boxes, rot_mat):
        centers, lengths = input_boxes[:, 0:3], input_boxes[:, 3:6]
        new_centers = np.dot(centers, np.transpose(rot_mat))

        dx, dy = lengths[:, 0] / 2.0, lengths[:, 1] / 2.0
        new_x = np.zeros((dx.shape[0], 4))
        new_y = np.zeros((dx.shape[0], 4))

        for i, crnr in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
            crnrs = np.zeros((dx.shape[0], 3))
            crnrs[:, 0] = crnr[0] * dx
            crnrs[:, 1] = crnr[1] * dy
            crnrs = np.dot(crnrs, np.transpose(rot_mat))
            new_x[:, i] = crnrs[:, 0]
            new_y[:, i] = crnrs[:, 1]

        new_dx = 2.0 * np.max(new_x, 1)
        new_dy = 2.0 * np.max(new_y, 1)
        new_lengths = np.stack((new_dx, new_dy, lengths[:, 2]), axis=1)

        return np.concatenate([new_centers, new_lengths], axis=1)


@DATASETS.register_module()
class Joint3DDataset_JointDC_v2c_nr3d_butd(torch.utils.data.Dataset):

    def __init__(self,
                split="train",
                data_root='/data/pointcloud/data_for_vote2cap/',
                transform=None,
                dataset_config=DatasetConfig(),
                num_points=50000,
                use_color=True,
                use_normal=False,
                use_multiview=False,
                use_height=False,
                augment=False,
                test_mode=False, 
                test_cfg=None, 
                loop=1
            ):

        self.dataset_config = dataset_config
        self.max_des_len = 32
        
        # initialize tokenizer and set tokenizer's `padding token` to `eos token`
        self.tokenizer = ScanReferTokenizer(SCANREFER['vocabulary']['word2idx'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        assert split in ["train", "val"]
        
        self.data_path = os.path.join(data_root, "data", "scannet", "scannet_data")

        all_scan_names = list(
            set(
                [
                    os.path.basename(x)[0:12]
                    for x in os.listdir(self.data_path)
                    if x.startswith("scene")
                ]
            )
        )
        if split in ["train", "val"]:
            
            self.scanrefer = SCANREFER['language'][split]
            self.scan_names = SCANREFER['scene_list'][split]
            # self.scan_names = self.scan_names[:10]  # debug

            # for joint training
            if split == "train":
                self.scan_names = SCANREFER['scene_list'][split] * 10

            self.split = split
            print(f"kept {len(self.scan_names)} scans out of {len(all_scan_names)}")
            
        else:
            raise ValueError(f"Unknown split name {split}")

        self.num_points = num_points
        self.use_color = use_color
        self.use_normal = use_normal
        self.use_multiview = use_multiview
        self.use_height = use_height
        self.augment = augment
        self.augment_det = False
        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]
        
        self.gathered_language = self.preprocess_and_gather_language()
        self.multiview_data = {}
        self.label_map = read_label_mapping(
            'pointcept/datasets/preprocessing/scanrefer/meta_data/scannetv2-labels.combined.tsv',
            label_from='raw_category',
            label_to='id'
        )

        self.scans = unpickle_data(f'/data/pointcloud/data_for_eda/scannet_others_processed/{split}_v3scans.pkl')
        self.scans = list(self.scans)[0]

        if os.path.exists('/home/lhj/lyd/3DRefTR/data/cls_results.json'):
            with open('/home/lhj/lyd/3DRefTR/data/cls_results.json') as fid:
                self.cls_results = json.load(fid)

    def default_dict_factory(self):
        return defaultdict(list)

    def preprocess_and_gather_language(self):
        
        gathered_language = defaultdict(self.default_dict_factory)
        
        for lang_dict in tqdm.tqdm(self.scanrefer):
            scene_id  = lang_dict['scene_id']
            object_id = int(lang_dict['object_id'])
            
            sentence  = ' '.join(lang_dict['token'] + [self.tokenizer.eos_token])
            gathered_language[scene_id][object_id].append(sentence)
        
        return gathered_language
    

    def _get_token_positive_map(self, captions, targets):
        """Return correspondence of boxes to tokens."""
        # Token start-end span in characters
        caption = ' '.join(captions.replace(',', ' ,').split())
        caption = ' ' + caption + ' '
        
        tokens_positive = np.zeros((MAX_NUM_OBJ, 2))
        tokenizer_roberta = \
            RobertaTokenizerFast.from_pretrained(
                '/data/pointcloud/data_for_eda/scannet_others_processed/roberta-base/', 
                local_files_only=True)
        
        if isinstance(targets, list):
            cat_names = targets
        else:
            cat_names = [targets]

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
        tokenized = tokenizer_roberta.batch_encode_plus(
            [' '.join(captions.replace(',', ' ,').split())],
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
    
    
    def __len__(self):
        return len(self.scan_names)
    

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

    # BRIEF GroupFree detection boxes
    def _get_detected_objects(self, split, scan_id, augmentations):
        # Initialize
        all_detected_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        all_detected_bbox_label_mask = np.array([False] * MAX_NUM_OBJ)
        detected_class_ids = np.zeros((MAX_NUM_OBJ,))
        detected_logits = np.zeros((MAX_NUM_OBJ, NUM_CLASSES))

        # Load: class, box, pc, logits
        detected_dict = np.load(
            f'/data/pointcloud/data_for_eda/scannet_others_processed/group_free_pred_bboxes/group_free_pred_bboxes_{split}/{scan_id}.npy',
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

            if augmentations.get('yz_flip', False):
                all_det_pts[:, 0] = -all_det_pts[:, 0]
            if augmentations.get('xz_flip', False):
                all_det_pts[:, 1] = -all_det_pts[:, 1]
            all_det_pts[:, 0:3] = np.dot(all_det_pts[:, 0:3], np.transpose(augmentations['rot_mat']))

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

    def __getitem__(self, idx):

        scan_name = self.scan_names[idx] # len(self.scan_names) 562
        mesh_vertices = np.load(
            os.path.join(self.data_path, scan_name) + "_aligned_vert.npy"
        )
        instance_labels = np.load(
            os.path.join(self.data_path, scan_name) + "_ins_label.npy"
        )
        semantic_labels = np.load(
            os.path.join(self.data_path, scan_name) + "_sem_label.npy"
        )
        instance_bboxes = np.load(
            os.path.join(self.data_path, scan_name) + "_aligned_bbox.npy"
        )

        if not self.use_color:
            point_cloud = mesh_vertices[:, 0:3]  # do not use color for now
            pcl_color = mesh_vertices[:, 3:6]
        else:
            point_cloud = mesh_vertices[:, 0:6]
            point_cloud[:, 3:] = (point_cloud[:, 3:] - MEAN_COLOR_RGB) / 256.0
            pcl_color = point_cloud[:, 3:]
        
        if self.use_normal:
            normals = mesh_vertices[:,6:9]
            point_cloud = np.concatenate([point_cloud, normals], 1)
        
        if self.use_multiview:
            # load multiview database
            pid = mp.current_process().pid
            if pid not in self.multiview_data:
                self.multiview_data[pid] = h5py.File(
                    os.path.join(self.data_path, 'enet_feats_maxpool.hdf5'), 
                    'r', libver='latest'
                )
            multiview = self.multiview_data[pid][scan_name]
            point_cloud = np.concatenate([point_cloud, multiview], 1)
        
        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)

        # ------------------------------- LABELS ------------------------------
        MAX_NUM_OBJ = self.dataset_config.max_num_obj
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6), dtype=np.float32)
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ), dtype=np.float32)
        raw_sizes = np.zeros((MAX_NUM_OBJ, 3), dtype=np.float32)
        object_ids = np.zeros((MAX_NUM_OBJ,))
        
        if self.split == "val":
            np.random.seed(1184)  # fix points in val and test
        point_cloud, choices = pc_util.random_sampling(
            point_cloud, self.num_points, return_choices=True
        )
        
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        pcl_color = pcl_color[choices]

        target_bboxes_mask[0 : instance_bboxes.shape[0]] = 1
        target_bboxes[0 : instance_bboxes.shape[0], :] = instance_bboxes[:, 0:6]

        # ------------------------------- DATA AUGMENTATION ------------------------------
        augmentations = {}
        if self.augment:
            
            augmentations['yz_flip'] = False
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                target_bboxes[:, 0] = -1 * target_bboxes[:, 0]
                augmentations['yz_flip'] = True

            augmentations['xz_flip'] = False
            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:, 1] = -1 * point_cloud[:, 1]
                target_bboxes[:, 1] = -1 * target_bboxes[:, 1]
                augmentations['xz_flip'] = True

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
            rot_mat = pc_util.rotz(rot_angle)
            augmentations['rot_mat'] = rot_mat
            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            target_bboxes = self.dataset_config.rotate_aligned_boxes(
                target_bboxes, rot_mat
            )

        raw_sizes = target_bboxes[:, 3:6]
        point_cloud_dims_min = point_cloud[..., :3].min(axis=0)
        point_cloud_dims_max = point_cloud[..., :3].max(axis=0)

        box_centers = target_bboxes.astype(np.float32)[:, 0:3]
        box_centers_normalized = shift_scale_points(
            box_centers[None, ...],
            src_range=[
                point_cloud_dims_min[None, ...],
                point_cloud_dims_max[None, ...],
            ],
            dst_range=self.center_normalizing_range,
        )
        box_centers_normalized = box_centers_normalized.squeeze(0)
        box_centers_normalized = box_centers_normalized * target_bboxes_mask[..., None]
        mult_factor = point_cloud_dims_max - point_cloud_dims_min
        box_sizes_normalized = scale_points(
            raw_sizes.astype(np.float32)[None, ...],
            mult_factor=1.0 / mult_factor[None, ...],
        )
        box_sizes_normalized = box_sizes_normalized.squeeze(0)

        # HACK: store the instance index
        object_ids[:instance_bboxes.shape[0]] = instance_bboxes[:, -1]
        
        captions = [
                'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
                'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub',
                'others'
            ]
        # captions = [
        #         'cabinet', 'bed', 'chair', 'couch', 'table', 'door',
        #         'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
        #         'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub',
        #         'other furniture'
        #     ]
        captions = ' . '.join(captions)

        targets = [self.dataset_config.class2type[
                    self.dataset_config.nyu40id2class[int(ind)]]
                        for ind in instance_bboxes[:, -2]] #scene0364_00 len(instance_bboxes) 11

        ret_dict = {}
        ret_dict["point_clouds"] = torch.from_numpy(point_cloud.astype(np.float32))  # [50000, 6]
        ret_dict["center_label"] = box_centers.astype(np.float32)  # [132, 3]

        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_bboxes_semcls[0 : instance_bboxes.shape[0]] = [
            self.dataset_config.nyu40id2class[x]
            for x in instance_bboxes[:, -2][0 : instance_bboxes.shape[0]]
        ]
        ret_dict["sem_cls_label"] = target_bboxes_semcls.astype(np.int64)
        ret_dict["box_label_mask"] = target_bboxes_mask.astype(np.float32)
        ret_dict["og_color"] = pcl_color.astype(np.float32)
        ret_dict["size_gts"] = raw_sizes.astype(np.float32)
        
        # HACK: add ground truth object ids
        ret_dict["gt_box_object_ids"] = object_ids.astype(np.int64)
        ret_dict["offset"] = torch.tensor([point_cloud.shape[0]])
        ret_dict["superpoint"] = torch.zeros((1))  # avoid bugs
        ret_dict["source_xzy"] = point_cloud[..., 0:3].astype(np.float32)
        ret_dict["utterances"] = (
                ' '.join(captions.replace(',', ' ,').split())
                + ' . not mentioned'
            )
        # ret_dict["utterances"] = "chair . not mentioned"  # caption ability

        (
            all_detected_bboxes, all_detected_bbox_label_mask,
            detected_class_ids, detected_logits
        ) = self._get_detected_objects(self.split, scan_name, augmentations)

        scan = self.scans[scan_name]
        (class_ids, all_bboxes, all_bbox_label_mask) = self._get_scene_objects(scan)

        all_detected_bboxes = all_bboxes
        all_detected_bbox_label_mask = all_bbox_label_mask
        detected_class_ids = np.zeros((len(all_bboxes,)))
        classes = np.array(self.cls_results[scan_name])
        detected_class_ids[all_bbox_label_mask] = classes[classes > -1]

        ret_dict["all_detected_boxes"] = all_detected_bboxes.astype(np.float32)
        ret_dict["all_detected_bbox_label_mask"] = all_detected_bbox_label_mask.astype(np.bool8)
        ret_dict["all_detected_class_ids"] = detected_class_ids.astype(np.int64)
        ret_dict["all_detected_logits"] = detected_logits.astype(np.float32)    

        tokens_positive, positive_map, modify_positive_map, pron_positive_map, \
                other_entity_map, auxi_entity_positive_map, rel_positive_map = self._get_token_positive_map(captions, targets)
        auxi_box = np.zeros((1, 6))

        ret_dict["tokens_positive"] = tokens_positive.astype(np.int64)
        ret_dict["positive_map"] = positive_map.astype(np.float32)
        ret_dict["modify_positive_map"] = modify_positive_map.astype(np.float32)
        ret_dict["pron_positive_map"] = pron_positive_map.astype(np.float32)
        ret_dict["other_entity_map" ]= other_entity_map.astype(np.float32)
        ret_dict["rel_positive_map"] = rel_positive_map.astype(np.float32)
        ret_dict["auxi_entity_positive_map"] = auxi_entity_positive_map.astype(np.float32)
        ret_dict["auxi_box"] = auxi_box.astype(np.float32)
        ret_dict["gt_masks"] = np.zeros((MAX_NUM_OBJ, 1))
        ret_dict["language_dataset"] = "scanrefer"
        ret_dict["point_instance_label"] = instance_labels.astype(np.int64)

        # caption label
        reference_tokens = np.zeros((MAX_NUM_OBJ, self.max_des_len))
        reference_masks  = np.zeros((MAX_NUM_OBJ, self.max_des_len))

        if self.split == 'train':
            
            scene_caption = []
            if scan_name in self.gathered_language:
                
                for instance_id in instance_bboxes[:, -1]:
                    if instance_id not in self.gathered_language[scan_name]:
                        caption = ''
                    else:
                        caption = random.choice(
                            self.gathered_language[scan_name][instance_id]
                        )
                    scene_caption.append(caption)
            
            tokenizer_output = self.tokenizer.batch_encode_plus(
                scene_caption, 
                max_length=self.max_des_len, 
                padding='max_length', 
                truncation='longest_first', 
                return_tensors='np'
            )
            tokenizer_output['input_ids'] *= tokenizer_output['attention_mask']
            
            reference_tokens[:len(instance_bboxes[:, -1])] = tokenizer_output['input_ids']
            reference_masks[:len(instance_bboxes[:, -1])]  = tokenizer_output['attention_mask']
            
        ret_dict['reference_tokens'] = reference_tokens.astype(np.int64)
        ret_dict['reference_masks'] = reference_masks.astype(np.float32)
        ret_dict["scan_idx"] = np.array(idx).astype(np.int64)
        
        return ret_dict
    

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

# if __name__ == '__main__':
#     dataset = Joint3DDataset_JointDC_v2c()
#     i = 0
#     while True:
#         print(i)
#         dataset.__getitem__(i)
#         i += 1
