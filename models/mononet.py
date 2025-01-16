import torch
import torch.nn as nn

from .encoder.gelan import GELAN
from .encoder.transformer import FeatureTransformer
from .decoder.keypoint_decoder import KeypointDecoder


class MonoNet(nn.Module):
    def __init__(self, cfg):
        super(MonoNet, self).__init__()

        feature_size = cfg.get('image_size')[0] // 16
        embedded_dim = cfg.model.get('embedded_dim')

        self.backbone = GELAN(gelan_type=cfg.model.get('backbone'))
        self.proj = nn.Conv2d(
            self.backbone.out_channels, embedded_dim, 1, bias=False)

        self.transformer = FeatureTransformer(
            embed_size=embedded_dim,
            feature_size=feature_size,
            nheads=cfg.model.get('num_heads'),
            nlayers=cfg.model.get('num_layers'))

        self.keypoint_decoder = KeypointDecoder(
            c_in=embedded_dim,
            c_out=cfg.dataset.get('num_joints'),
            num_layers=2)

    def forward(self, x):
        """
        xs(list): size is (batch_size, 3, 256, 256)
        proj_list(list): (batch_size, 3, 4)
        """
        # extract features using cnn backbone
        feats = self.backbone(x)
        feats = self.proj(feats)

        # enhance features using transformer encoder
        feats, attn = self.transformer(feats)

        # decode features to heatmaps
        heatmaps = self.keypoint_decoder(feats)

        return heatmaps, attn


def build_mono_net(cfg):
    return MonoNet(cfg)


if __name__ == '__main__':
    model = MonoNet(
        num_joints=17,
        image_size=[256, 256],
        backbone='gelanS',
        embedded_dim=256,
        num_heads=8,
        num_layers=4
    )
    x = torch.randn(32, 3, 256, 256)
    heatmaps, attn = model(x)
    print(heatmaps.shape, attn.shape)
