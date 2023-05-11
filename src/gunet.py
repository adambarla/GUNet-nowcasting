import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_guided_filter.guided_filter import FastGuidedFilter


class GUNet(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate, r, epsilon):
        super().__init__()

        features = 2 ** math.ceil(math.log2(in_channels) + 1)

        # Encoder # (256, 512)
        self.encoder1 = GUNet._block(in_channels, features, dropout_rate=dropout_rate)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # (128, 256)
        self.encoder2 = GUNet._block(features, features * 2, dropout_rate=dropout_rate)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # (64, 128)
        self.encoder3 = GUNet._block(features * 2, features * 4, dropout_rate=dropout_rate)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # (32, 64)
        self.encoder4 = GUNet._block(features * 4, features * 8, dropout_rate=dropout_rate)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # (16, 32)
        self.encoder5 = GUNet._block(features * 8, features * 16, dropout_rate=dropout_rate)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # (8, 16)

        # Bottleneck
        self.bottleneck = GUNet._block(features * 16, features * 32, dropout_rate=dropout_rate)

        # Decoder
        self.fgif5 = FastGuidedFilter(r, features * 32, features * 16, eps=epsilon)
        self.decoder5 = GUNet._block(features * 16, features * 16, dropout_rate=dropout_rate)
        self.fgif4 = FastGuidedFilter(r, features * 16, features * 8, eps=epsilon)
        self.decoder4 = GUNet._block(features * 8, features * 8, dropout_rate=dropout_rate)
        self.fgif3 = FastGuidedFilter(r, features * 8, features * 4, eps=epsilon)
        self.decoder3 = GUNet._block(features * 4, features * 4, dropout_rate=dropout_rate)
        self.fgif2 = FastGuidedFilter(r, features * 4, features * 2, eps=epsilon)
        self.decoder2 = GUNet._block(features * 2, features * 2, dropout_rate=dropout_rate)
        self.fgif1 = FastGuidedFilter(r, features * 2, features, eps=epsilon)
        self.decoder1 = GUNet._block(features, features)

        # Output
        self.output = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))
        # enc6 = self.encoder6(self.pool5(enc5))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool5(enc5))

        # Decoder
        dec5 = self.fgif5(bottleneck, bottleneck, enc5)
        dec5 = self.decoder5(dec5)
        dec4 = self.fgif4(enc5, dec5, enc4)
        dec4 = self.decoder4(dec4)
        dec3 = self.fgif3(enc4, dec4, enc3)
        dec3 = self.decoder3(dec3)
        dec2 = self.fgif2(enc3, dec3, enc2)
        dec2 = self.decoder2(dec2)
        dec1 = self.fgif1(enc2, dec2, enc1)
        dec1 = self.decoder1(dec1)

        # Output
        output = self.output(dec1)
        return self.activation(output)

    @staticmethod
    def _block(in_channels, features, dropout_rate=0.5):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
