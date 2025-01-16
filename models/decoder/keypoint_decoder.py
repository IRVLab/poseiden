import torch.nn as nn


class KeypointDecoder(nn.Module):
    def __init__(self, c_in, c_out, num_layers):
        super(KeypointDecoder, self).__init__()

        self.deconv_layers = self._make_deconv_layer(c_in, c_in//2, num_layers)
        self.final_layer = nn.Conv2d(
            in_channels=c_in//2,
            out_channels=c_out,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def _get_deconv_layer(self, c_in, c_out, kernel_size, stride, padding):
        return [nn.ConvTranspose2d(
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=0,
                    bias=False),
                nn.BatchNorm2d(c_out, momentum=0.1),
                nn.ReLU(inplace=True)]

    def _make_deconv_layer(self, c_in, c_out, num_layers):
        layers = []

        for _ in range(num_layers):
            layers.extend(self._get_deconv_layer(c_in, c_out, 4, 2, 1))
            c_in = c_out

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x

    def init_weights(self):
        for m in self.deconv_layers.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
