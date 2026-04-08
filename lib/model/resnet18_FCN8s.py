from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn.functional as F
import os
import logging
import functools
import numpy as np
import torch
import torch.nn as nn

# import our stuffs
from .bn_helper import BatchNorm2d, BatchNorm2d_class, relu_inplace
from .resnet18_new_32 import Resnet18_new_32


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class UNETblock(nn.Module):
    def __init__(self, lowchannle, highchnnle):
        super(UNETblock, self).__init__()
        self.high_conv=ConvBNReLU(highchnnle,lowchannle,3,1,1)


    def forward(self, low, high):
        b,c_low,h_low,w_low=low.size()
        high=F.interpolate(high, [h_low,w_low], mode='bilinear', align_corners=True)
        high=self.high_conv(high)
        # fcat = torch.cat([high, low], dim=1)
        fsum=high+low
        return fsum


class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=BatchNorm2d, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=BatchNorm2d, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )


    def forward(self, x):
        return self.block(x)


class Seg_Resnet18_unet(nn.Module):
    def __init__(self,nclass):
        super(Seg_Resnet18_unet, self).__init__()
        self.nclass = nclass
        self.pretrained = Resnet18_new_32()
        self.score1=nn.Sequential(nn.Conv2d(128, nclass, 1))
        self.score2=nn.Sequential(nn.Conv2d(256, nclass, 1))
        self.score3 =_FCNHead(512, nclass, BatchNorm2d)
        self.aux_head = nn.Conv2d(256, nclass, 1)


    def base_forward(self, x):
        x, feat8, feat16, feat32 = self.pretrained(x)
        return x, feat8, feat16, feat32


    def forward(self, x):
        _, _, h, w = x.size()
        c1, c2, c3, c4 = self.base_forward(x)
        aux_fuse = self.aux_head(c3)
        c4_score=self.score3(c4)
        c3_score=self.score2(c3)
        c2_score=self.score1(c2)
        upscore3 = F.interpolate(c4_score, c3_score.size()[2:], mode='bilinear', align_corners=True)
        fuse=upscore3+c3_score
        fuse = F.interpolate(fuse, c2_score.size()[2:], mode='bilinear', align_corners=True)
        fuse = fuse + c2_score
        fuse = F.interpolate(fuse,(h,w), mode='bilinear', align_corners=True)
        return aux_fuse, fuse


def get_seg_model(cfg, **kwargs):
    model = Seg_Resnet18_unet(cfg.DATASET.NUM_CLASSES, **kwargs)
    return model