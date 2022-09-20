import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU(inplace=True)
    )
    return layer

def conv_class():
    layer = nn.Sequential(
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        
    )
    return layer


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),####
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),####
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ClassConv(nn.Module):
    """(convolution => [BN] => ReLU) """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.pooling=nn.MaxPool2d(16,16)
        self.line = nn.Linear(in_channels,out_channels,bias=False)

    def forward(self, x):
        x1=self.pooling(x)
        x1=x1.view(x1.size(0),-1)
        x2=self.line(x1)
        return x2

class scSE(nn.Module):
    def __init__(self, in_channels):
        super(scSE, self).__init__()
        #cSE
        self.cSE_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),# Global Average Pooling
            nn.Conv2d(in_channels, in_channels//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, in_channels, kernel_size=1), 
            nn.Sigmoid()
        )
        #sSE
        self.sSE_block = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        #cSE
        x1 = self.cSE_block(x)
        x2 = x * x1
        #sSE
        x3 = self.sSE_block(x)
        x4 = x * x3
        #scSE
        x5 = x4 + x2
        return x5


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.se1 = scSE(64)
        self.down1 = Down(64, 128)
        self.se2 = scSE(128)
        self.down2 = Down(128, 256)
        self.se3 = scSE(256)
        self.down3 = Down(256, 512)
        self.se4 = scSE(512)
        self.down4 = Down(512, 1024)
       
        self.up1 = Up(1024, 512, bilinear)
        self.se5 = scSE(512)
        self.up2 = Up(512, 256, bilinear)
        self.se6 = scSE(256)
        self.up3 = Up(256, 128, bilinear)
        self.se7 = scSE(128)
        self.up4 = Up(128, 64, bilinear)
        self.se8 = scSE(64)

        self.outconv = OutConv(64, 2)
        self.outs = OutConv(64, n_classes)

        self.outc = conv_class()

    def forward(self, x):
        x1 = self.inc(x)
        x11 = self.se1(x1)
        x2 = self.down1(x11)
        x21 = self.se2(x2)
        x3 = self.down2(x21)
        x31 = self.se3(x3)
        x4 = self.down3(x31)
        x41 = self.se4(x4)
        x5 = self.down4(x41)
        x = self.up1(x5, x41)
        x = self.se5(x)
        x = self.up2(x, x31)
        x = self.se6(x)
        x = self.up3(x, x21)
        x = self.se7(x)
        x = self.up4(x, x11)
        x = self.se8(x) #4 64 256 256
        feature1 = x
        feature2 = x
        seg_result = self.outs(x)
        x = self.outconv(x)
        feature3 = x
        cls_result = self.outc(x)
        return seg_result, cls_result

class FPSNet(nn.Module):
    def __init__(self):
        super(FPSNet, self).__init__()
        self.net = UNet(n_channels=3, n_classes=1)
        
    def forward(self, x):
        seg_result, cls_result = self.net(x)
        return  seg_result, cls_result

 