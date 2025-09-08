import torch 
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool3d(2)

        self.bottleneck = DoubleConv(512, 1024) # bottleneck layer (last)

        self.upconv4 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2) # upsampling
        self.dec4 = DoubleConv(1024, 512) # double convolution 
        self.upconv3 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.conv_last = nn.Conv3d(64, out_channels, kernel_size=1) # last conv layer

    def forward(self, x):
        enc1 = self.enc1(x) # 64
        enc2 = self.enc2(self.pool(enc1)) #128
        enc3 = self.enc3(self.pool(enc2)) #256
        enc4 = self.enc4(self.pool(enc3)) #512  

        bottleneck = self.bottleneck(self.pool(enc4)) #1024

        dec4 = self.upconv4(bottleneck) # upsample
        dec4 = torch.cat((dec4, enc4), dim=1) # concatenate with skip connection
        dec4 = self.dec4(dec4) # double conv

        dec3 = self.upconv3(dec4) # upsample
        dec3 = torch.cat((dec3, enc3), dim=1) # concatenate with skip connection
        dec3 = self.dec3(dec3) # double conv

        dec2 = self.upconv2(dec3) # upsample
        dec2 = torch.cat((dec2, enc2), dim=1) # concatenate with skip connection
        dec2 = self.dec2(dec2) # double conv

        dec1 = self.upconv1(dec2) # upsample
        dec1 = torch.cat((dec1, enc1), dim=1) # concatenate with skip connection
        dec1 = self.dec1(dec1) # double conv 

        return self.conv_last(dec1) # final conv layer to get desired output channels


