from cv2 import preCornerDetect
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ClassConv(nn.Module):
   
    def __init__(self):
        super().__init__()

        self.pooling = nn.AvgPool2d(kernel_size=16,stride=16,padding=0)
        self.conv = nn.Conv2d(64, 32, kernel_size=3)
        self.relu = nn.ReLU(inplace=True)

        self.line1 = nn.Linear(2048,300,bias=False) #4608
        self.line2 = nn.Linear(300,250,bias=False)
        self.line3 = nn.Linear(250,200,bias=False)
        self.line4 = nn.Linear(200,150,bias=False)
        self.line5 = nn.Linear(150,100,bias=False)
        self.line6 = nn.Linear(400,50,bias=False)
        self.line7 = nn.Linear(50,2,bias=False)

    def forward(self, x):
        x = self.pooling(x)
        x = self.conv(x)
        x = self.relu(x)

        x = x.view(x.size(0),-1)
        x1 = self.line1(x)
        x1 = self.relu(x1)
        x2 = self.line2(x1)
        x2 = self.relu(x2)
        x3 = self.line3(x2)
        x3 = self.relu(x3)
        x4 = self.line4(x3)
        x4 = self.relu(x4)
        x5 = self.line5(x4)
        x5 = self.relu(x5)
        x5 = torch.cat([x1, x5], dim=1)
        x6 = self.line6(x5)
        x6 = self.relu(x6)
        x7 = self.line7(x6)
    
        return x7

class Net(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        self.outs = OutConv(64, n_classes)
        self.outc = ClassConv()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
       
        pred_seg = self.outs(x)
        pred_class = self.outc(x)
        
        return pred_seg, pred_class
    
net = Net(n_channels=3, n_classes=1) 

 