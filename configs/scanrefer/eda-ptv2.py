_base_ = ["../_base_/default_runtime.py"]
# misc custom setting
batch_size = 64 # bs: total bs in all gpus  80
mix_prob = 0.8
enable_amp = True
num_worker = 4
batch_size_val = 16
batch_size_test = 16
eval_freq = 3
find_unused_parameters = True

# model settings
model = dict(
    type="DefaultGrounder",
    backbone=dict(
        type="eda_ptv2_dets3d",
        butd=False  # not used butd
    ),
    losses=['boxes', 'labels', 'contrastive_align']
)

# scheduler settings
epoch = 100
eval_epoch = 100
optimizer = dict(type="AdamW", lr=2e-4, weight_decay=0.0005)
scheduler = dict(type="MultiStepLR", gamma=0.1, milestones=[0.5, 0.75])

# dataset settings
dataset_type = "Joint3DDataset"
data_root = "/userhome/backup_lhj/dataset/pointcloud/data_for_eda/scannet_others_processed"

data = dict(
    num_classes=13,
    ignore_index=-1,
    names=["ceiling", "floor", "wall", "beam", "column", "window", "door",
           "table", "chair", "sofa", "bookcase", "board", "clutter"],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1/64, 1/64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1/64, 1/64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train",
                 keys=("coord", "color", "segment"), return_discrete_coord=True),
            dict(type="SphereCrop", point_max=100000, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "discrete_coord", "segment"), feat_keys=["coord", "color"])
        ],
        test_mode=False
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train",
                 keys=("coord", "color", "segment"), return_discrete_coord=True),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect",
                 keys=("coord", "discrete_coord", "segment"), feat_keys=["coord", "color"])
        ],
        test_mode=False),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor")
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color"),
                return_discrete_coord=True),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "discrete_coord", "index"),
                    feat_keys=("coord", "color"))
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.1, 1.1])],
                [dict(type="RandomScale", scale=[0.9, 0.9]),
                 dict(type="RandomFlip", p=1)],
                [dict(type="RandomScale", scale=[0.95, 0.95]),
                 dict(type="RandomFlip", p=1)],
                [dict(type="RandomScale", scale=[1, 1]),
                 dict(type="RandomFlip", p=1)],
                [dict(type="RandomScale", scale=[1.05, 1.05]),
                 dict(type="RandomFlip", p=1)],
                [dict(type="RandomScale", scale=[1.1, 1.1]),
                 dict(type="RandomFlip", p=1)],
            ]
        )
    )
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="GroundingEvaluator", losses=['boxes', 'labels', 'contrastive_align']),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False)
]

# tester
test = dict(
    type="GroundingTester",
    losses=['boxes', 'labels', 'contrastive_align']
)
