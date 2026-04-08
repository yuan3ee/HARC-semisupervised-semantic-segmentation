import math, time
from itertools import chain
import torch
import torch.nn.functional as F
from torch import nn
# import our stuffs
from .decoders_CCT import *
from .encoder import Encoder


class CCT(nn.Module):

    def __init__(self, cfg, num_classes=6, testing=True, pretrained=True):
        super(CCT, self).__init__()
        self.encoder = Encoder(pretrained=pretrained)
        self.testing = testing
        # The main encoder
        upscale = 8
        num_out_ch = 2048
        decoder_in_ch = num_out_ch // 4
        num_classes = cfg.DATASET.NUM_CLASSES
        self.main_decoder = MainDecoder(upscale, decoder_in_ch, num_classes=num_classes)


    def forward(self, x_l=None, target_l=None, x_ul=None, target_ul=None, curr_iter=None, epoch=None):
        if self.testing:
            return self.main_decoder(self.encoder(x_l))


    def get_backbone_params(self):
        return self.encoder.get_backbone_params()


def get_seg_model(cfg, **kwargs):
    model = CCT(cfg)
    return model