import os
import torch
import torch.nn as nn
from collections import OrderedDict
from einops import rearrange
from thop import profile, clever_format

from .encoder.gelan import GELAN
from .encoder.transformer import FeatureTransformer
from .decoder.disparity_decoder import DisparityDecoder
from .decoder.keypoint_decoder import KeypointDecoder


class StereoNet(nn.Module):
    def __init__(self, cfg):
        super(StereoNet, self).__init__()

        feature_size = cfg.get('image_size')[0] // 16
        heatmap_size = cfg.get('image_size')[0] // 4
        embedded_dim = cfg.model.get('embedded_dim')

        self.backbone = GELAN(gelan_type=cfg.model.get('backbone'))
        self.proj = nn.Conv2d(
            self.backbone.out_channels, embedded_dim, 1, bias=False)

        self.transformer = FeatureTransformer(
            embed_size=embedded_dim,
            feature_size=feature_size,
            nheads=cfg.model.get('num_heads'),
            nlayers=cfg.model.get('num_layers'))

        self.disparity_decoder = DisparityDecoder(
            c_in=embedded_dim * 2,
            c_out=cfg.dataset.get('num_joints'),
            heatmap_size=heatmap_size,
            dmax=cfg.model.get('dmax'),
            dmin=cfg.model.get('dmin'),
            num_disp=cfg.model.get('num_disp'))

        self.keypoint_decoder = KeypointDecoder(
            c_in=embedded_dim * 2,
            c_out=cfg.dataset.get('num_joints'),
            num_layers=2)

    def init_weights(self, pretrained='', device='cuda'):
        if os.path.isfile(pretrained):
            # load pretrained model
            checkpoint = torch.load(
                pretrained, map_location=device)["state_dict"]

            # only load weights from encoders
            state_dict = OrderedDict()
            for key in checkpoint.keys():
                if key.startswith('model.backbone'):
                    state_dict[key.replace("model.", "")] = checkpoint[key]
                if key.startswith('model.proj'):
                    state_dict[key.replace("model.", "")] = checkpoint[key]
                if key.startswith('model.transformer'):
                    state_dict[key.replace("model.", "")] = checkpoint[key]

            self.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError("Pretrained model '{}' does not exist."
                             .format(pretrained))

    def process_heatmap(self, heatmap):
        """
        This function computes the 2D location of each joint by integrating
        heatmaps across spatial axes.
        That is, the 2D location of each joint j represents the center of mass
        of the jth feature map.
        Args:
            heatmap (batch_size, num_joints, N, N): heatmap features
        Returns:
            cxy (batch_size, num_joints, 2): 2D locations of the joints
        """
        b, j, h, w = heatmap.size()

        # Perform softmax along spatial axes.
        heatmap = rearrange(heatmap, 'b j h w -> b j (h w)')
        heatmap = nn.functional.softmax(heatmap, dim=2)
        heatmap = rearrange(heatmap, 'b j (h w) -> b j h w', h=h, w=w)

        # Compute 2D locations of the joints as the center of mass of the
        # corresponding heatmaps.
        x = torch.arange(w, dtype=torch.float, device=heatmap.device)
        y = torch.arange(h, dtype=torch.float, device=heatmap.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')

        cx = torch.sum(grid_x * heatmap, dim=[2, 3])
        cy = torch.sum(grid_y * heatmap, dim=[2, 3])

        cxy = torch.stack([cx, cy], dim=-1)

        return cxy

    def forward(self, xl, xr):
        """
        xs(list): size is (batch_size, 3, 256, 256)
        proj_list(list): (batch_size, 3, 4)
        """
        img_size = xl.size(2)

        # extract features using cnn backbone
        fl, fr = self.backbone(xl), self.backbone(xr)
        fl, fr = self.proj(fl), self.proj(fr)

        # fuse features using transformer encoder
        fl, fr, attn = self.transformer(fl, fr)

        x = torch.cat((fl, fr), dim=1)

        # decode features to heatmaps and disparity
        kps_2d = self.keypoint_decoder(x)
        disp = self.disparity_decoder(x)

        heatmap_size = kps_2d.size(2)
        kps_2d = self.process_heatmap(kps_2d)
        kps_2d = kps_2d * (img_size / heatmap_size)

        kps_2ds = [kps_2d, kps_2d.clone()]  # [left_view, right_view]
        kps_2ds[0][:, :, :1] = kps_2ds[0][:, :, :1] + disp / 2
        kps_2ds[1][:, :, :1] = kps_2ds[1][:, :, :1] - disp / 2

        return kps_2ds, disp, attn


def build_stereo_net(cfg):
    model = StereoNet(cfg)

    image_size = cfg.get("image_size")
    input_ = torch.randn(1, 3, image_size[0], image_size[1])
    flops, params = profile(model, inputs=(input_, input_))
    flops, params = clever_format([flops, params], "%.3f")
    print(f"Params: {params}, FLOPs: {flops}")
    return model


if __name__ == '__main__':
    model = StereoNet(
        num_joints=17,
        image_size=[256, 256],
        backbone='gelanS',
        embedded_dim=256,
        num_heads=8,
        num_layers=4,
        dmax=30,
        dmin=5,
        num_disp=30
    )
    xs = [torch.randn(32, 3, 256, 256) for _ in range(2)]
    kps_2d, disp, attn = model(xs[0], xs[1])
    print(kps_2d[0].shape, kps_2d[1].shape)
    print(disp.shape, attn[0].shape)
