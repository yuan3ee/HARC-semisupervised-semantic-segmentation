
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()
_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.GPUS = (0, 1)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME_D = 'discriminator_DUL'
_C.MODEL.PRETRAINED_D = ''

_C.MODEL.NAME = 'seg_hrnet'
_C.MODEL.PRETRAINED = ''
_C.MODEL.ALIGN_CORNERS = True
_C.MODEL.NUM_OUTPUTS = 1
_C.MODEL.STUDENT = ''
_C.MODEL.STUDENT_PRETRAINED = ''


_C.MODEL.EXTRA = CN(new_allowed=True)


_C.MODEL.OCR = CN()
_C.MODEL.OCR.MID_CHANNELS = 512
_C.MODEL.OCR.KEY_CHANNELS = 256
_C.MODEL.OCR.DROPOUT = 0.05
_C.MODEL.OCR.SCALE = 1


_C.LOSS = CN()
# _C.LOSS.LAMBDA_FM = 0.1
_C.LOSS.DYNAMIC = False

_C.LOSS.USE_OHEM = False
_C.LOSS.OHEMTHRES = 0.9
_C.LOSS.OHEMKEEP = 100000
_C.LOSS.CLASS_BALANCE = False
_C.LOSS.BALANCE_WEIGHTS = [1.0]


# DATASET related params
_C.DATASET = CN()
_C.DATASET.LABELED_RATIO = 0.25
_C.DATASET.TRAINSIZE = 0

_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'cityscapes'
_C.DATASET.NUM_CLASSES = 19
_C.DATASET.TRAIN_SET = 'list/cityscapes/train.lst'
_C.DATASET.EXTRA_TRAIN_SET = ''
_C.DATASET.TEST_SET = 'list/cityscapes/val.lst'


# VAT
_C.VAT = CN()
_C.VAT.ENABLE = False
_C.VAT.XI = 1e-6
_C.VAT.EPS = 2.0


# DropOut
_C.DROPOUT = CN()
_C.DROPOUT.ENABLE = False
_C.DROPOUT.XI = 0.5
_C.DROPOUT.EPS = True


# cutout
_C.CUTOUT = CN()
_C.CUTOUT.ENABLE = False
_C.CUTOUT.ERASE = 0.4


# ContextMasking
_C.CONTEXTMASKING = CN()
_C.CONTEXTMASKING.ENABLE = False


# ObjectMasking
_C.OBJECTMASKING = CN()
_C.OBJECTMASKING.ENABLE = False


# FeatureDrop
_C.FEATUREDROP = CN()
_C.FEATUREDROP.ENABLE = False


# FeatureNoise
_C.FEATURENOISE = CN()
_C.FEATURENOISE.ENABLE = False
_C.FEATURENOISE.UNIFORM_RANGE = 0.3


# The temporary test
_C.TEMP = CN()
_C.TEMP.INSTRUCTIONS = ''
_C.TEMP.OTHER = [1.0, 0.0]


# training
_C.TRAIN = CN()
# _C.TRAIN.TRAIN_D_STEPS = 1
_C.TRAIN.LR_D = 1e-4
_C.TRAIN.THRESHOLD_ST = 1.0
_C.TRAIN.THRESHOLD_ST_ANTI = 0.0
# _C.TRAIN.THRESHOLD_SOFT_ST = 0.0
# _C.TRAIN.THRESHOLD_SOFT_ST_ANTI = 1.0
_C.TRAIN.ST_BEGIN_EPOCH = 30
_C.TRAIN.ST_AGAIN_EPOCH = 150
_C.TRAIN.ST_AF = 0.3
_C.TRAIN.VALID_PER_EPOCH = 1
# _C.TRAIN.ST_FOR_DEEPSP = False
_C.TRAIN.CCT = False
_C.TRAIN.FULL = False
_C.TRAIN.USE_BEST_MODEL = False

_C.TRAIN.FREEZE_LAYERS = ''
_C.TRAIN.FREEZE_EPOCHS = -1
_C.TRAIN.NONBACKBONE_KEYWORDS = []
_C.TRAIN.NONBACKBONE_MULT = 10
_C.TRAIN.IMAGE_SIZE = [1024, 512]  # width * height
_C.TRAIN.BASE_SIZE = 2048
_C.TRAIN.DOWNSAMPLERATE = 1
_C.TRAIN.FLIP = True
_C.TRAIN.MULTI_SCALE = True
_C.TRAIN.SCALE_FACTOR = 16
_C.TRAIN.RANDOM_BRIGHTNESS = False
_C.TRAIN.RANDOM_BRIGHTNESS_SHIFT_VALUE = 10
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.01
_C.TRAIN.EXTRA_LR = 0.001
_C.TRAIN.OPTIMIZER = 'sgd'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.IGNORE_LABEL = -1
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 484
_C.TRAIN.EXTRA_EPOCH = 0
_C.TRAIN.RESUME = False
_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True
# only using some training samples
_C.TRAIN.NUM_SAMPLES = 0


# testing
_C.TEST = CN()
_C.TEST.MODEL_D_FILE = ''

_C.TEST.IMAGE_SIZE = [2048, 1024]  # width * height
_C.TEST.BASE_SIZE = 2048
_C.TEST.BATCH_SIZE_PER_GPU = 32
# only testing some samples
_C.TEST.NUM_SAMPLES = 0
_C.TEST.MODEL_FILE = ''
_C.TEST.FLIP_TEST = False
_C.TEST.MULTI_SCALE = False
_C.TEST.SCALE_LIST = [1]
_C.TEST.OUTPUT_INDEX = -1


# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False


# hung model
_C.HUNG = CN()
_C.HUNG.ENABLED = False
_C.HUNG.LAMBDA_SEMI = 0.1
_C.HUNG.LAMBDA_SEMI_ADV = 0.001
_C.HUNG.SEMI_START_ADV = 0
_C.HUNG.LAMBDA_ADV_PRED = 0.1
_C.HUNG.D_REMAIN = True

# s4GAN model
_C.S4GAN = CN()
_C.S4GAN.ENABLED = False
_C.S4GAN.LAMBDA_FM = 0.1
_C.S4GAN.LAMBDA_ST = 1.0


# SEMI_ALL
_C.SEMI_ALL = CN()
_C.SEMI_ALL.ENABLED = False


def update_config(cfg, args):
    cfg.defrost()
    
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

