# _*_ coding: utf-8 _*_
"""
Time:     2020/11/22 下午3:25
Author:   Cheng Ding(Deeachain)
File:     UNet.py
Describe: Write during my study in Nanjing University of Information and Secience Technology
Github:   https://github.com/Deeachain
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bn_helper import BatchNorm2d, BatchNorm2d_class, relu_inplace
from .resnet import ResNetBackbone


__all__ = ["UNet"]

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class conv3x3_block_x1(nn.Module):
    '''(conv => BN => ReLU) * 1'''

    def __init__(self, in_ch, out_ch):
        super(conv3x3_block_x1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv3x3_block_x2(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(conv3x3_block_x2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(upsample, self).__init__()
        self.conv1x1 = conv1x1(in_ch, out_ch)
        self.conv = conv3x3_block_x2(in_ch, out_ch)

    def forward(self, H, L):
        """
        H: High level feature map, upsample
        L: Low level feature map, block output
        """
        H = F.interpolate(H, scale_factor=2, mode='bilinear', align_corners=False)
        H = self.conv1x1(H)
        x = torch.cat([H, L], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, cfg):
        super(UNet, self).__init__()
        self.pretrained = ResNetBackbone(cfg)(arch = 'resnet50')
        self.maxpool = nn.MaxPool2d(2)
        self.bottle = self.pretrained.resinit
        self.pool = self.pretrained.maxpool
        self.block1 = self.pretrained.layer1 # conv3x3_block_x2(3, 64)
        self.block2 = self.pretrained.layer2 # conv3x3_block_x2(64, 128)
        self.block3 = self.pretrained.layer3 # conv3x3_block_x2(128, 256)
        self.block4 = self.pretrained.layer4 # conv3x3_block_x2(256, 512)
        self.block_out = conv3x3_block_x1(2048, 4096)
        self.upsample1 = upsample(4096, 2048)
        self.upsample2 = upsample(2048, 1024)
        self.upsample3 = upsample(1024, 512)
        self.upsample4 = upsample(512, 256)
        self.upsample_out = conv3x3_block_x2(256, cfg.DATASET.NUM_CLASSES)

        # self._init_weight()

    def forward(self, x):
        x = self.pretrained.resinit(x)
        x = self.maxpool(x)
        block1_x = self.block1(x)
        block2_x = self.block2(block1_x)
        block3_x = self.block3(block2_x)
        block4_x = self.block4(block3_x)
        x = self.maxpool(block4_x)
        # print(block1_x.size())  # torch.Size([2, 256, 128, 128])
        # print(block2_x.size())  # torch.Size([2, 512, 64, 64])
        # print(block3_x.size())  # torch.Size([2, 1024, 32, 32])
        # print(block4_x.size())  # torch.Size([2, 2048, 16, 16])
        x = self.block_out(x)
        x = self.upsample1(x, block4_x)
        x = self.upsample2(x, block3_x)
        x = self.upsample3(x, block2_x)
        x = self.upsample4(x, block1_x)
        x = self.upsample_out(x)

        return x


def get_seg_model(cfg, **kwargs):
    model = UNet(cfg)
    return model