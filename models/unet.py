import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Contracting Path (Encoder)
        self.enc_conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding='same')
        self.dropout1 = nn.Dropout(0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding='same')
        self.dropout2 = nn.Dropout(0.1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        self.dropout3 = nn.Dropout(0.2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc_conv4 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.dropout4 = nn.Dropout(0.2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding='same')
        self.dropout5 = nn.Dropout(0.3)
        self.bottleneck_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding='same')
        
        # Expansive Path (Decoder)
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding='same')
        self.dropout6 = nn.Dropout(0.2)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding='same')
        self.dropout7 = nn.Dropout(0.2)
        
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_conv3 = nn.Conv2d(64, 32, kernel_size=3, padding='same')
        self.dropout8 = nn.Dropout(0.1)
        
        self.upconv4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec_conv4 = nn.Conv2d(32, 16, kernel_size=3, padding='same')
        self.dropout9 = nn.Dropout(0.1)
        
        # Output Layer
        self.output_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = F.relu(self.enc_conv1(x))
        c1 = self.dropout1(c1)
        p1 = self.pool1(c1)
        
        c2 = F.relu(self.enc_conv2(p1))
        c2 = self.dropout2(c2)
        p2 = self.pool2(c2)
        
        c3 = F.relu(self.enc_conv3(p2))
        c3 = self.dropout3(c3)
        p3 = self.pool3(c3)
        
        c4 = F.relu(self.enc_conv4(p3))
        c4 = self.dropout4(c4)
        p4 = self.pool4(c4)
        
        # Bottleneck
        b1 = F.relu(self.bottleneck_conv1(p4))
        b1 = self.dropout5(b1)
        b2 = F.relu(self.bottleneck_conv2(b1))
        
        # Decoder
        u1 = self.upconv1(b2)
        u1 = torch.cat((u1, c4), dim=1)
        c5 = F.relu(self.dec_conv1(u1))
        c5 = self.dropout6(c5)
        
        u2 = self.upconv2(c5)
        u2 = torch.cat((u2, c3), dim=1)
        c6 = F.relu(self.dec_conv2(u2))
        c6 = self.dropout7(c6)
        
        u3 = self.upconv3(c6)
        u3 = torch.cat((u3, c2), dim=1)
        c7 = F.relu(self.dec_conv3(u3))
        c7 = self.dropout8(c7)
        
        u4 = self.upconv4(c7)
        u4 = torch.cat((u4, c1), dim=1)
        c8 = F.relu(self.dec_conv4(u4))
        c8 = self.dropout9(c8)
        
        # Output
        outputs = self.output_conv(c8)
        return outputs
