import torch.nn as nn

from .gelan import GELAN
from .transformer import FeatureTransformer


class FeaturesExtractor(nn.Module):
    def __init__(self, backbone, embedded_dim=256,
                 feature_size=16, num_heads=8, num_layers=4):
        super(FeaturesExtractor, self).__init__()
        backbone_dict = {
            "gelanS": {
                "network": GELAN("small"),
                "output_channels": 512
            },
            "gelanL": {
                "network": GELAN("large"),
                "output_channels": 512
            }
        }

        self.backbone = backbone_dict[backbone]["network"]
        out_channels = backbone_dict[backbone]["output_channels"]

        self.proj = nn.Conv2d(out_channels, embedded_dim, 1, bias=False)
        self.out_channels = out_channels

        self.transformer = FeatureTransformer(
            embedded_dim, feature_size, num_heads, num_layers)

    def forward(self, xl, xr):
        # extract features using cnn backbone
        fl, fr = self.backbone(xl), self.backbone(xr)
        fl, fr = self.proj(fl), self.proj(fr)

        # exchange features using self and cross attention
        fl, fr, attn = self.transformer(fl, fr)

        return fl, fr, attn
