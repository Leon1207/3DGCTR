
This is a PyTorch implementation of 3DRefTR proposed by our paper [Rethinking 3D Dense Caption and Visual Grounding in A Unified Framework through Prompt-based Localization](https://arxiv.org/abs/2404.11064).

## 0. Install

We train and evaluate our model using CUDA 11.3.
```
conda create -n pointcept113 python=3.8 -y
conda activate pointcept113
conda install ninja -y
# Choose version you want here: https://pytorch.org/get-started/previous-versions/
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric

conda install pytorch-scatter -c pyg
pip install spacy
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.3.0/en_core_web_sm-3.3.0.tar.gz

sudo apt-get install make gcc g++ -y
sudo apt-get install manpages-dev -y
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --no-deps
```

## 1. Data preparation

Please refer to [3DRefTR](https://github.com/Leon1207/3DRefTR) and [Vote2Cap-DETR](https://github.com/ch3cook-fdu/Vote2Cap-DETR) to download 3DVG and 3DDC datasets for training and evaluation, then set your [dataset path] in the config file.

## 2. Train & Evaluation

- Pretrain on 3DVG
```
sh scripts/train.sh -p python -g 4 -d scanrefer -c ScanRefer_3DVG_pretrain -n [NAME]
```
- Joint training on 3DVG and 3DDC
```
sh scripts/train.sh -p python -g 4 -d scanrefer -c ScanRefer_3DVG_3DDC_joint_mle -n [NAME]
```
- SCST training on 3DDC
```
sh scripts/train.sh -p python -g 4 -d scanrefer -c ScanRefer_3DDC_scst -n [NAME]
```

- Evaluation: you can change the tester in the config file
```
test = dict(type="CaptionTester")  # GroundingTester
```
```
sh scripts/test.sh -p python -d scanrefer -n [NAME] -w model_best
```

## 3. Visulization

You can refer to [this repo](https://github.com/yigengjiang/3DGCTR-Visualization) for visualization.


## Citation
If you find _3DGCTR_ useful to your research, please cite our work:
```
@misc{luo2024rethinking3ddensecaption,
      title={Rethinking 3D Dense Caption and Visual Grounding in A Unified Framework through Prompt-based Localization}, 
      author={Yongdong Luo and Haojia Lin and Xiawu Zheng and Yigeng Jiang and Fei Chao and Jie Hu and Guannan Jiang and Songan Zhang and Rongrong Ji},
      year={2024},
      eprint={2404.11064},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.11064}, 
}
```

## Acknowledgement

This repository is built on reusing codes of [Pointcept](https://github.com/Pointcept/Pointcept). We recommend using their code repository in your research and reading the related article.
