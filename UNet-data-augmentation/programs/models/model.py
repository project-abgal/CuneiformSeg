import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .parts import double_conv, down, outconv, inconv, up
import torchvision.models as models

# reference: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py


class UNet(nn.Module):
    """U-Net."""

    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256, bilinear)
        self.up2 = up(512, 128, bilinear)
        self.up3 = up(256, 64, bilinear)
        self.up4 = up(128, 64, bilinear)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x_1 = self.inc(x)
        x_2 = self.down1(x_1)
        x_3 = self.down2(x_2)
        x_4 = self.down3(x_3)
        x_5 = self.down4(x_4)
        x = self.up1(x_5, x_4)
        x = self.up2(x, x_3)
        x = self.up3(x, x_2)
        x = self.up4(x, x_1)
        x = self.outc(x)
        return F.tanh(x)


class MiniUNet(nn.Module):
    """Reduced U-Net."""

    def __init__(self, n_channels, n_classes, bilinear=True):
        super(MiniUNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 256)
        #  self.down4 = down(512, 512)
        #  self.up1 = up(1024, 256, bilinear)
        self.up2 = up(512, 128, bilinear)
        self.up3 = up(256, 64, bilinear)
        self.up4 = up(128, 64, bilinear)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x_1 = self.inc(x)
        x_2 = self.down1(x_1)
        x_3 = self.down2(x_2)
        x_4 = self.down3(x_3)
        #  x_5 = self.down4(x_4)
        #  x = self.up1(x_5, x_4)
        x = self.up2(x_4, x_3)
        x = self.up3(x, x_2)
        x = self.up4(x, x_1)
        x = self.outc(x)
        return F.tanh(x)


class one_color_vgg16(nn.Module):
    """
    there is a fatal bug. Do not use.
    """

    def __init__(self, n_channels, n_classes):
        super(one_color_vgg16, self).__init__()
        self.base_net = models.vgg16_bn()
        self.base_layers = list(self.base_net.children())
        self.layer0 = nn.Conv2d(n_channels, 64, kernel_size=(3, 3))
        self.layerother = nn.Sequential(*self.base_layers[1:])

    def forward(self, input):
        x = self.layer0(input)
        x = self.layerother(x)
        return F.sigmoid(x)


class one_color_resnet(nn.Module):

    def __init__(self, n_channels, n_classes):
        super(one_color_resnet, self).__init__()
        self.base_net = models.resnet18()
        self.base_layers = list(self.base_net.children())
        self.layer0 = nn.Conv2d(n_channels, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.layermiddle = nn.Sequential(*self.base_layers[1:-1])
        self.layerlast = nn.Linear(512, n_classes, bias=True) # have to be changed depending on crop size

    def forward(self, input):
        x = self.layer0(input)
        x = self.layermiddle(x)
        x = F.leaky_relu_(self.layerlast(x.reshape(x.shape[0], -1)))
        return F.sigmoid(x)
