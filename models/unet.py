import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def conv_block(in_channels, out_channels, use_dropout=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels)
    ]
    if use_dropout:
        layers.append(nn.Dropout(0.1))
    return nn.Sequential(*layers)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Downsampling path
        self.dconv_down1 = conv_block(3, 32)
        self.dconv_down2 = conv_block(32, 64)
        self.dconv_down3 = conv_block(64, 128)
        self.dconv_down4 = conv_block(128, 256) 
        
        self.maxpool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(256, 512, use_dropout=True)
        
        # Upsampling path
        self.upsample4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dconv_up4 = conv_block(512, 256)
        self.upsample3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dconv_up3 = conv_block(256, 128)
        self.upsample2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dconv_up2 = conv_block(128, 64)
        self.upsample1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dconv_up1 = conv_block(64, 32)
        
        self.conv_last = nn.Conv2d(32, 1, 1) 
        
    def forward(self, x):
        # Downsampling
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Upsampling
        x = self.upsample4(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.dconv_up4(x)

        x = self.upsample3(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        
        x = self.upsample2(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        
        x = self.upsample1(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        
        # Final convolution
        out = self.conv_last(x)
        
        return out



def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias.data, 0)

def init_unet_model(device):
    model = UNet().to(device)
    model.apply(initialize_weights)
    model.to(device)
    
    return model
