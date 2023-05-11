import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate, k):
        super().__init__()

        features = 2 ** math.ceil(math.log2(in_channels) + 1)

        # Encoder # (256, 512)
        self.encoder1 = UNet._block(in_channels, features, dropout_rate=dropout_rate)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # (128, 256)
        self.encoder2 = UNet._block(features, features * 2, dropout_rate=dropout_rate)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # (64, 128)
        self.encoder3 = UNet._block(features * 2, features * 4, dropout_rate=dropout_rate)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # (32, 64)
        self.encoder4 = UNet._block(features * 4, features * 8, dropout_rate=dropout_rate)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # (16, 32)
        self.encoder5 = UNet._block(features * 8, features * 16, dropout_rate=dropout_rate)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # (8, 16)

        # Bottleneck
        self.bottleneck = UNet._block(features * 16, features * 32, dropout_rate=dropout_rate)

        padding = (k - 2) // 2
        # Decoder
        self.upconv5 = nn.ConvTranspose2d(features * 32, features * 16, kernel_size=k, stride=2, padding=padding)
        self.decoder5 = UNet._block(features * 32, features * 16, dropout_rate=dropout_rate)
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=k, stride=2, padding=padding)
        self.decoder4 = UNet._block(features * 16, features * 8, dropout_rate=dropout_rate)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=k, stride=2, padding=padding)
        self.decoder3 = UNet._block(features * 8, features * 4, dropout_rate=dropout_rate)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=k, stride=2, padding=padding)
        self.decoder2 = UNet._block(features * 4, features * 2, dropout_rate=dropout_rate)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=k, stride=2, padding=padding)
        self.decoder1 = UNet._block(features * 2, features)

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

        # Bottleneck
        bottleneck = self.bottleneck(self.pool5(enc5))

        # Decoder
        dec5 = self.upconv5(bottleneck)
        dec5 = self.decoder5(torch.cat((dec5, enc5), 1))
        dec4 = self.upconv4(dec5)
        dec4 = self.decoder4(torch.cat((dec4, enc4), 1))
        dec3 = self.upconv3(dec4)
        dec3 = self.decoder3(torch.cat((dec3, enc3), 1))
        dec2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat((dec2, enc2), 1))
        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat((dec1, enc1), 1))

        # Output
        output = self.output(dec1)
        return self.activation(output)

    @staticmethod
    def _block(in_channels, out_channels, dropout_rate=0.5):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

