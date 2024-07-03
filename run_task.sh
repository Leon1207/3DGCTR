# conda init bash
source /opt/conda/etc/profile.d/conda.sh
conda activate pointcept102 
cd /home/lhj/lyd/VL-Pointcept/

# scst
# sh scripts/train.sh -p python -g 1 -d scanrefer -c eda-s3d-dc-v2c-scst-cross -n eda-dc-v2ctraining-joint10-cross-smalllr2-scst5
# nr3d-vg
# sh scripts/train.sh -p python -g 4 -d scanrefer -c semseg-3dreftr-ptv2maxpool-v2c-nr3d -n 3dreftr_sp_ptv2maxpool_coord1024_nobutd_nr3d
# nr3d-vg-decay
# sh scripts/train.sh -p python -g 4 -d scanrefer -c semseg-3dreftr-ptv2maxpool-v2c-nr3d-decay -n 3dreftr_sp_ptv2maxpool_coord1024_nobutd_nr3d_decay
# s3d-pretrain
# sh scripts/train.sh -p python -g 4 -d scanrefer -c eda-s3d-pretrain -n eda-s3d-pretrain-view
# nr3d-joint
# sh scripts/train.sh -p python -g 4 -d scanrefer -c eda-s3d-dc-v2c-joint-cross-smalllr-nr3d -n eda-dc-v2ctraining-joint10-cross-smalllr2-nr3d2
# nr3d-scst
# sh scripts/train.sh -p python -g 1 -d scanrefer -c eda-s3d-dc-v2c-scst-cross-nr3d -n eda-dc-v2ctraining-joint10-cross-smalllr2-scst3-nr3d
# eda-rec-res
# sh scripts/train.sh -p python -g 4 -d scanrefer -c semseg-3dreftr-ptv2maxpool-v2c-eda -n 3dreftr-sp-v2c-nobutd
# eda-mle
# sh scripts/train.sh -p python -g 4 -d scanrefer -c eda-s3d-dc-v2c-joint-cross-smalllr-eda -n eda-dc-v2ctraining-joint10-cross-smalllr2-eda2
# eda-mle-butd
sh scripts/train.sh -p python -g 4 -d scanrefer -c eda-s3d-dc-v2c-joint-cross-smalllr-eda-butd -n eda-dc-v2ctraining-joint10-cross-smalllr2-eda-butd
