"""
SparseUNet Driven by SpConv (recommend)

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

import spconv.pytorch as spconv
from torch_geometric.utils import scatter

from timm.models.layers import trunc_normal_

from ..utils import offset2batch, batch2offset
import pointops
import numpy as np


class BasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self,
                 in_channels,
                 embed_channels,
                 stride=1,
                 norm_fn=None,
                 indice_key=None,
                 bias=False,
                 ):
        super().__init__()

        assert norm_fn is not None

        if in_channels == embed_channels:
            self.proj = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.proj = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, embed_channels, kernel_size=1, bias=False),
                norm_fn(embed_channels)
            )

        self.conv1 = spconv.SubMConv3d(
            in_channels, embed_channels, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(embed_channels)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            embed_channels, embed_channels, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(embed_channels)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        out = out.replace_feature(out.features + self.proj(residual).features)
        out = out.replace_feature(self.relu(out.features))

        return out


class SpUNetBase(nn.Module):
    def __init__(self,
                 in_channels,
                 base_channels=32,
                #  channels=(32, 64, 128, 256, 256, 128, 96, 288),
                #  layers=(2, 3, 4, 6, 2, 2, 2, 2),
                 channels=(32, 64, 128, 256, 256, 288, 1, 1),
                 layers=(2, 3, 4, 6, 2, 2, 1, 1),
                 cls_mode=False):
        super().__init__()
        assert len(layers) % 2 == 0
        assert len(layers) == len(channels)
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channels = channels
        self.layers = layers
        self.num_stages = len(layers) // 2
        self.cls_mode = cls_mode
        self.grid_size = 0.02
        self.hashing = True

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        block = BasicBlock

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, base_channels, kernel_size=5, padding=1, bias=False, indice_key='stem'),
            norm_fn(base_channels),
            nn.ReLU(),
        )

        enc_channels = base_channels
        dec_channels = channels[-1]
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList() if not self.cls_mode else None

        for s in range(self.num_stages):
            # encode num_stages
            self.down.append(spconv.SparseSequential(
                spconv.SparseConv3d(enc_channels, channels[s], kernel_size=2, stride=2, bias=False,
                                    indice_key=f"spconv{s + 1}"),
                norm_fn(channels[s]),
                nn.ReLU()
            ))
            self.enc.append(spconv.SparseSequential(OrderedDict([
                # (f"block{i}", block(enc_channels, channels[s], norm_fn=norm_fn, indice_key=f"subm{s + 1}"))
                # if i == 0 else
                (f"block{i}", block(channels[s], channels[s], norm_fn=norm_fn, indice_key=f"subm{s + 1}"))
                for i in range(layers[s])
            ])))
            if not self.cls_mode:
                # decode num_stages
                self.up.append(spconv.SparseSequential(
                    spconv.SparseInverseConv3d(channels[len(channels) - s - 2], dec_channels,
                                               kernel_size=2, bias=False, indice_key=f"spconv{s + 1}"),
                    norm_fn(dec_channels),
                    nn.ReLU()
                ))
                self.dec.append(spconv.SparseSequential(OrderedDict([
                    (f"block{i}",
                     block(dec_channels + enc_channels, dec_channels, norm_fn=norm_fn, indice_key=f"subm{s}"))
                    if i == 0 else
                    (f"block{i}", block(dec_channels, dec_channels, norm_fn=norm_fn, indice_key=f"subm{s}"))
                    for i in range(layers[len(channels) - s - 1])
                ])))

            enc_channels = channels[s]
            dec_channels = channels[len(channels) - s - 2]

        final_in_channels = channels[-1] if not self.cls_mode else channels[self.num_stages - 1]
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def hash(self, arr):
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr

    def forward(self, pointcloud, offset, bss):
        discrete_coord = torch.floor(pointcloud[..., 0:3].contiguous() / self.grid_size).int()
        discrete_coord_min = discrete_coord.min(0).values
        discrete_coord -= discrete_coord_min
        source_offset = offset

        # hash avoid duplicate
        if self.hashing:
            feat = []
            offset = []
            dc = []
            discrete_coord = discrete_coord.view(bss, -1, 3)
            features = pointcloud[..., 3:].view(bss, -1, 3)
            for bs in range(bss):
                key = self.hash(discrete_coord[bs].cpu().numpy())
                idx_sort = np.argsort(key)
                key_sort = key[idx_sort]
                _, _, count = np.unique(key_sort, return_inverse=True, return_counts=True)
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
                idx_unique = idx_sort[idx_select]
                if len(feat) == 0:
                    feat = features[bs][idx_unique]
                    offset = torch.tensor([len(idx_unique)], device=feat.device)
                    dc = discrete_coord[bs][idx_unique]
                else:
                    feat = torch.cat([feat, features[bs][idx_unique]], dim=0)
                    offset = torch.cat([offset, torch.tensor([len(idx_unique)], device=feat.device) + offset[bs-1].item()], dim=0)
                    dc = torch.cat([dc, discrete_coord[bs][idx_unique]], dim=0)
            discrete_coord = dc

        # not using hash
        else:
            feat = pointcloud[..., 3:]
            offset = offset.int()
        
        batch = offset2batch(offset)
        sparse_shape = torch.add(torch.max(discrete_coord, dim=0).values, 96).tolist()
        x = spconv.SparseConvTensor(
            features=feat,
            indices=torch.cat([batch.unsqueeze(-1).int(), discrete_coord.int()], dim=1).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=batch[-1].tolist() + 1
        )
        x = self.conv_input(x)
        skips = [x]
        # enc forward
        for s in range(self.num_stages):
            x = self.down[s](x)
            x = self.enc[s](x)
            skips.append(x)
        x = skips.pop(-1)
        enc_num = 0  # only use first two layers of decoder
        if not self.cls_mode:
            # dec forward
            for s in reversed(range(self.num_stages)):
                x = self.up[s](x)
                skip = skips.pop(-1)
                x = x.replace_feature(torch.cat((x.features, skip.features), dim=1))
                x = self.dec[s](x)
                enc_num += 1
                if enc_num == 2:
                    break

        # fps sample 1024 points, then assign features by ball query
        fps_num = 1024
        coord = pointcloud[..., 0:3].contiguous()
        down_xyz = (x.indices[..., 1:] + discrete_coord_min) * self.grid_size
        feats, down_offset = x.features, batch2offset(x.indices[..., 0])
        new_offset = torch.tensor([(b + 1) * fps_num for b in range(bss)]).cuda()
        fps_inds = pointops.farthest_point_sampling(coord, source_offset, new_offset)
        xyz = coord[fps_inds.long(), :]
        grouped_feature, _ = pointops.ball_query_and_group(
            feat=feats,
            xyz=down_xyz,
            offset=down_offset,
            new_xyz=xyz,
            new_offset=new_offset,
            max_radio=0.2,
            nsample=2)
        # grouped_feature, _ = pointops.knn_query_and_group(
        #     feat=feats,
        #     xyz=down_xyz,
        #     offset=down_offset,
        #     new_xyz=xyz,
        #     new_offset=new_offset,
        #     nsample=1)
        grouped_feature = grouped_feature.max(1)[0]

        end_points = {}
        end_points['fp2_features'] = grouped_feature.view(bss, fps_num, 288).transpose(-1, -2).contiguous()
        end_points['fp2_xyz'] = xyz.view(bss, fps_num, 3).float()
        
        fps_inds = list(torch.split(fps_inds, fps_num, dim=0))
        for b in range(bss - 1):
            fps_inds[b + 1] = fps_inds[b + 1] - source_offset[b]
        fps_inds = torch.stack(fps_inds, dim=0)
        end_points['fp2_inds'] = fps_inds

        return end_points