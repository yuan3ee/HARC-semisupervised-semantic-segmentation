# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# CCT
import lib.models.decoders

# discriminator
import lib.models.discriminator_s4GAN
import lib.models.discriminator_DUL
import lib.models.discriminator_hung

# generator
import lib.models.hrnetOCR
import lib.models.deeplabv2_syncBn
import lib.models.resnet18_FCN8s
import lib.models.resnet50_FCN8s
import lib.models.resnet50_unet
import lib.models.unet
import lib.models.model
import lib.models.CCT_fcn
import lib.models.resnet50_FCN8s_RegionContrast
import lib.models.resnet50_FCN8s_contrast_ly_noregist
# backbone
import lib.models.resnet18_new_32
import lib.models.resnet
