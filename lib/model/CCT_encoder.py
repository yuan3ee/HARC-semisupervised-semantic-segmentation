import torch
import torch.nn as nn
import torch.nn.functional as F
import os
# import our stuffs
from .backbones.resnet_backbone import ResNetBackbone


resnet50 = {
    "path": "models/backbones/pretrained/3x3resnet50-imagenet.pth",
}


class _FCNHead(nn.Module):

    def __init__(self, in_channels, channels, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):

    def __init__(self, pretrained):
        super(Encoder, self).__init__()
        self.innerC = 512
        self.pretrained = ResNetBackbone(backbone='resnet50', pretrained=pretrained)
        self.base = nn.Sequential(
            nn.Sequential(self.pretrained.prefix, self.pretrained.maxpool),
            self.pretrained.layer1,
            self.pretrained.layer2,
            self.pretrained.layer3,
            self.pretrained.layer4
        )

        self.score1=nn.Sequential(nn.Conv2d(512, self.innerC, 1))
        self.score2=nn.Sequential(nn.Conv2d(1024, self.innerC, 1))
        self.score3 =_FCNHead(2048, self.innerC)
        self._fcn = nn.ModuleList([self.score1, self.score2, self.score3])
        # self.aux_head = nn.Conv2d(1024, self.innerC, 1)

    def base_forward(self, x):
        x = self.base[1](self.base[0](x))
        feat8 = self.base[2](x)
        feat16 = self.base[3](feat8)
        feat32 = self.base[4](feat16)

        return x, feat8, feat16, feat32

    def forward(self, x):
        _, _, h, w = x.size()
        c1, c2, c3, c4 = self.base_forward(x)
        c4_score = self.score3(c4)
        c3_score = self.score2(c3)
        c2_score = self.score1(c2)
        upscore3 = F.interpolate(c4_score, c3_score.size()[2:], mode='bilinear', align_corners=True)
        fuse = upscore3 + c3_score
        fuse = F.interpolate(fuse, c2_score.size()[2:], mode='bilinear', align_corners=True)
        fuse = fuse + c2_score
        return fuse

    def get_backbone_params(self):
        return self.base.parameters()

    def get_module_params(self):
        return self._fcn.parameters()