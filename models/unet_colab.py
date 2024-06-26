import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if dropout_prob > 0:
            self.dropout = nn.Dropout(dropout_prob)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.down1 = ConvBlock(in_channels, 16, dropout_prob=0.1)
        self.down2 = ConvBlock(16, 32, dropout_prob=0.1)
        self.down3 = ConvBlock(32, 64, dropout_prob=0.2)
        self.down4 = ConvBlock(64, 128, dropout_prob=0.2)
        self.down5 = ConvBlock(128, 256, dropout_prob=0.3)

        self.up1 = ConvBlock(256 + 128, 128, dropout_prob=0.2)
        self.up2 = ConvBlock(128 + 64, 64, dropout_prob=0.2)
        self.up3 = ConvBlock(64 + 32, 32, dropout_prob=0.1)
        self.up4 = ConvBlock(32 + 16, 16, dropout_prob=0.1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Contracting Path
        c1 = self.down1(x)
        p1 = self.maxpool(c1)

        c2 = self.down2(p1)
        p2 = self.maxpool(c2)

        c3 = self.down3(p2)
        p3 = self.maxpool(c3)

        c4 = self.down4(p3)
        p4 = self.maxpool(c4)

        c5 = self.down5(p4)

        # Expansive Path
        u6 = self.upsample(c5)
        u6 = torch.cat((u6, c4), dim=1)
        c6 = self.up1(u6)

        u7 = self.upsample(c6)
        u7 = torch.cat((u7, c3), dim=1)
        c7 = self.up2(u7)

        u8 = self.upsample(c7)
        u8 = torch.cat((u8, c2), dim=1)
        c8 = self.up3(u8)

        u9 = self.upsample(c8)
        u9 = torch.cat((u9, c1), dim=1)
        c9 = self.up4(u9)

        outputs = self.final_conv(c9)
        outputs = self.sigmoid(outputs)
        return outputs


def init_unet_model_colab(device):
    # model = UNet(in_channels=3, out_channels=1)
    model = UNet(in_channels=3, out_channels=1)
    
    model.to(device)

    return model