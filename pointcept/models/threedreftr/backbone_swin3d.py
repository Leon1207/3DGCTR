import torch
import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensor
from timm.models.layers import trunc_normal_

from pointcept.models.swin3d.mink_layers import MinkConvBNRelu, MinkResBlock
from pointcept.models.swin3d.swin3d_layers import GridDownsample, GridKNNDownsample, BasicLayer, Upsample
from pointcept.models.utils import offset2batch, batch2offset

import pointops


class Swin3DUNet(nn.Module):
    def __init__(self,
                 in_channels,
                 base_grid_size,
                 depths,
                 channels,
                 num_heads,
                 window_sizes,
                 quant_size,
                 drop_path_rate=0.2,
                 up_k=3,
                 num_layers=5,
                 stem_transformer=True,
                 down_stride=3,
                 upsample='linear_attn',
                 knn_down=True,
                 cRSE='XYZ_RGB_NORM',
                 fp16_mode=1):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        if knn_down:
            downsample = GridKNNDownsample
        else:
            downsample = GridDownsample

        self.cRSE = cRSE
        if stem_transformer:
            self.stem_layer = MinkConvBNRelu(
                in_channels=in_channels,
                out_channels=channels[0],
                kernel_size=3,
                stride=1,
            )
            self.layer_start = 0
        else:
            self.stem_layer = nn.Sequential(
                MinkConvBNRelu(
                    in_channels=in_channels,
                    out_channels=channels[0],
                    kernel_size=3,
                    stride=1,
                ),
                MinkResBlock(
                    in_channels=channels[0],
                    out_channels=channels[0]
                )
            )
            self.downsample = downsample(
                channels[0],
                channels[1],
                kernel_size=down_stride,
                stride=down_stride
            )
            self.layer_start = 1
        self.layers = nn.ModuleList([
            BasicLayer(
                dim=channels[i],
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_sizes[i],
                quant_size=quant_size,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample if i < num_layers - 1 else None,
                down_stride=down_stride if i == 0 else 2,
                out_channels=channels[i + 1] if i < num_layers - 1 else None,
                cRSE=cRSE,
                fp16_mode=fp16_mode) for i in range(self.layer_start, num_layers)])

        if 'attn' in upsample:
            up_attn = True
        else:
            up_attn = False

        self.upsamples = nn.ModuleList([
            Upsample(channels[i], channels[i - 1], num_heads[i - 1], window_sizes[i - 1], quant_size, attn=up_attn, \
                     up_k=up_k, cRSE=cRSE, fp16_mode=fp16_mode)
            for i in range(num_layers - 1, 0, -1)])
        
        self.final = nn.Sequential(
            nn.Linear(channels[0], 96),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.Linear(96, 288)
        )

        self.base_grid_size = base_grid_size
        self.init_weights()

    def forward(self, pointcloud, offset, bss):

        discrete_coord = torch.floor(pointcloud[..., 0:3].contiguous() / 0.02).int()
        discrete_coord_min = discrete_coord.min(0).values
        discrete_coord -= discrete_coord_min
        feat = pointcloud[..., 3:]
        coord_feat = feat  # feat not contains coords, so equal
        coord = pointcloud[..., 0:3].contiguous()
        offset = offset.int()

        batch = offset2batch(offset)
        in_field = ME.TensorField(
            features=torch.cat([batch.unsqueeze(-1),  # 1
                                coord / self.base_grid_size,  # 3
                                coord_feat / 1.001,  # 3
                                feat  # 3
                                ], dim=1),  # [n, 10]
            coordinates=torch.cat([batch.unsqueeze(-1).int(), discrete_coord.int()], dim=1),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=feat.device)

        sp = in_field.sparse()
        coords_sp = SparseTensor(
            features=sp.F[:, :coord_feat.shape[-1] + 4],
            coordinate_map_key=sp.coordinate_map_key,
            coordinate_manager=sp.coordinate_manager,
        )  # [n, 7], batch + coord + feat
        sp = SparseTensor(
            features=sp.F[:, coord_feat.shape[-1] + 4:],
            coordinate_map_key=sp.coordinate_map_key,
            coordinate_manager=sp.coordinate_manager,
        )  # [n, 3], feat
        sp_stack = []
        coords_sp_stack = []
        sp = self.stem_layer(sp)  # [n, 48]
        if self.layer_start > 0:
            sp_stack.append(sp)
            coords_sp_stack.append(coords_sp)
            sp, coords_sp = self.downsample(sp, coords_sp)

        for i, layer in enumerate(self.layers):
            coords_sp_stack.append(coords_sp)
            sp, sp_down, coords_sp = layer(sp, coords_sp)  # appear nan?
            sp_stack.append(sp)
            assert (coords_sp.C == sp_down.C).all()
            sp = sp_down

        sp = sp_stack.pop()
        coords_sp = coords_sp_stack.pop()
        dec_num = 0
        for i, upsample in enumerate(self.upsamples):
            sp_i = sp_stack.pop()
            coords_sp_i = coords_sp_stack.pop()
            sp = upsample(sp, coords_sp, sp_i, coords_sp_i)
            coords_sp = coords_sp_i
            dec_num += 1
            if dec_num == 2:
                break
        
        # fps sample 1024 points, then assign features by ball query
        final_feat = self.final(sp.slice(in_field).F)  # [n, 288]
        fps_num = 1024
        coord = pointcloud[..., 0:3].contiguous()
        down_xyz, feats, down_offset = coord, final_feat, offset
        new_offset = torch.tensor([(b + 1) * fps_num for b in range(bss)]).cuda()
        fps_inds = pointops.farthest_point_sampling(coord, offset, new_offset)
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
            fps_inds[b + 1] = fps_inds[b + 1] - offset[b]
        fps_inds = torch.stack(fps_inds, dim=0)
        end_points['fp2_inds'] = fps_inds

        return end_points

    def init_weights(self):
        """Initialize the weights in backbone.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)
