import logging
import pprint
import argparse
import os
import sys
import random
import timeit
import cv2
import numpy as np
import pickle
import scipy.misc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transform
from torch.utils import model_zoo
from torch.autograd import Variable
from itertools import chain
from tqdm import tqdm
# import our stuffs
from lib import models
from lib import data
import sys
from lib.utils.utils import *
from lib.config import config
from lib.config import update_config
from lib.models.resnet50_FCN8s_RegionContrast import RegionContrast
# from lib.models.resnet50_FCN8s_contrast_new import region_memory, region_contrast_loss
from lib.data.postdam_strong import postdam_augstrong_cutmix
from lib.models.resnet50_FCN8s_contrast_ly_noregist import region_memory, update_region_memory, computer_region_center, region_contrast_loss

import pickle

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")

    parser.add_argument('--cfg',
                        # default='/home/hnu/LY/semi_ly_1008/experiments/postdam/SEMI_contast_cutmix.yaml',
                        default = '/home/hnu/LY/semi_ly_1008/experiments/vaihingen/SEMI_contast_cutmix_wa.yaml',
                        help='experiment configure file name',
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--random-seed",
                        type=int,
                        default=1234,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=0)
    parser.add_argument("--contrast_weight",
                        type=float,
                        default=0.05)
    # parser.add_argument('contrast_loss_input',
    #                     default='')

    args = parser.parse_args()
    update_config(config, args)

    return args

start = timeit.default_timer()
args = get_arguments()
criterion = nn.BCELoss()

def main():
    # assert config.TRAIN.FULL is True

    if args.random_seed > 0:
        print('Seeding with', args.random_seed)
        random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    logger, rootOutputDir = createLogger(args, config, phase='train')

    if args.local_rank <= 0:
        logger.info(args)
        logger.info(config)
        logger.info(config.TEMP.INSTRUCTIONS)

    # init cuda
    gpus = list(config.GPUS)
    cudnn.benchmark = config.CUDNN.BENCHMARK  # TODO:設置False，判斷加速時長设置
    cudnn.deterministic = config.CUDNN.DETERMINISTIC  # TODO:设置True，避免波动，相当于设置CUDNN.BENCHMARK为False
    cudnn.enabled = config.CUDNN.ENABLED  # TODO:配合CUDNN.BENCHMARK为True使用

    # load data
    input_size = (config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1])
    dst = eval('data.' + str(config.DATASET.DATASET))
    batch_size = 4
    if config.DATASET.LABELED_RATIO is not None:
        partial_size_remain = int(config.DATASET.LABELED_RATIO * config.DATASET.TRAINSIZE)
        train_dataset = dst(config.DATASET.ROOT,
                           # '/home/hnu/LY/dataset/potsdam/512/train_1_8.lst',
                           '/home/hnu/LY/dataset/vaihingen/train_1_8.lst',
                           downsample_rate=config.TRAIN.DOWNSAMPLERATE,
                           base_size=config.TRAIN.BASE_SIZE,
                           crop_size=input_size,
                           scale=config.TRAIN.MULTI_SCALE,
                           mirror=config.TRAIN.FLIP,
                           ignore_label=config.TRAIN.IGNORE_LABEL,
                           scale_factor=config.TRAIN.SCALE_FACTOR)
        train_dataset_unlabel = postdam_augstrong_cutmix(config.DATASET.ROOT,
                                    config.DATASET.TRAIN_SET,
                                    downsample_rate=config.TRAIN.DOWNSAMPLERATE,
                                    num_samples=partial_size_remain,
                                    base_size=config.TRAIN.BASE_SIZE,
                                    crop_size=input_size,
                                    scale=config.TRAIN.MULTI_SCALE,
                                    mirror=config.TRAIN.FLIP,
                                    ignore_label=config.TRAIN.IGNORE_LABEL,
                                    scale_factor=config.TRAIN.SCALE_FACTOR)
    else:
        train_dataset = dst(config.DATASET.ROOT,
                            # '/home/hnu/LY/dataset/potsdam/512/train_1_8.lst',
                            '/home/hnu/LY/dataset/vaihingen/train_1_8.lst',
                            downsample_rate=config.TRAIN.DOWNSAMPLERATE,
                            base_size=config.TRAIN.BASE_SIZE,
                            crop_size=input_size,
                            scale=config.TRAIN.MULTI_SCALE,
                            mirror=config.TRAIN.FLIP,
                            ignore_label=config.TRAIN.IGNORE_LABEL,
                            scale_factor=config.TRAIN.SCALE_FACTOR)
        train_dataset_unlabel = postdam_augstrong_cutmix(config.DATASET.ROOT,
                                    config.DATASET.TRAIN_SET,
                                    downsample_rate=config.TRAIN.DOWNSAMPLERATE,
                                    base_size=config.TRAIN.BASE_SIZE,
                                    crop_size=input_size,
                                    scale=config.TRAIN.MULTI_SCALE,
                                    mirror=config.TRAIN.FLIP,
                                    ignore_label=config.TRAIN.IGNORE_LABEL,
                                    scale_factor=config.TRAIN.SCALE_FACTOR)
    trainloader_gt = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=batch_size, shuffle=True,
                                                 num_workers=config.WORKERS, pin_memory=True)
    trainloader_unlabel = torch.utils.data.DataLoader(train_dataset_unlabel,
                                                      batch_size=batch_size,
                                                      shuffle=True, num_workers=config.WORKERS,
                                                      pin_memory=True)


    test_size = (config.TEST.IMAGE_SIZE[0], config.TEST.IMAGE_SIZE[1])
    test_dataset = dst(config.DATASET.ROOT,
                       config.DATASET.TEST_SET,
                       downsample_rate=config.TRAIN.DOWNSAMPLERATE,
                       base_size=config.TEST.BASE_SIZE,
                       crop_size=test_size,
                       scale=config.TEST.MULTI_SCALE,
                       mirror=config.TEST.FLIP_TEST,
                       ignore_label=config.TRAIN.IGNORE_LABEL,
                       scale_factor=config.TRAIN.SCALE_FACTOR)
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,#batch_size,
        shuffle=False,
        num_workers= config.WORKERS,
        pin_memory=True)

    # init Student



    model = eval('models.' + config.MODEL.NAME + '.get_seg_model')(config)
    # print('model', 'models.' + config.MODEL.NAME + '.get_seg_model')

    # optimizer for segmentation network
    params_list = []
    for module in zip(model.parameters()):
        params_list.append(
            # dict(params=module.parameters(), lr=config.TRAIN.LR)
            dict(params=module, lr=config.TRAIN.LR)
        )

    optimizer = torch.optim.SGD(params_list,
                            lr=config.TRAIN.LR,
                            momentum=config.TRAIN.MOMENTUM,
                            weight_decay=config.TRAIN.WD,
                            nesterov=config.TRAIN.NESTEROV,  # TODO:设置为True
                            )


    optimizer.zero_grad()
    model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count())).cuda()
    model.train()

    # init Teacher
    model_teacher = eval('models.' + config.MODEL.NAME + '.get_seg_model')(config)
    model_teacher = nn.DataParallel(model_teacher, device_ids=range(torch.cuda.device_count())).cuda()
    for p in model_teacher.parameters():
        p.requires_grad = False

    with torch.no_grad():
        for t_params, s_params in zip(model_teacher.parameters(), model.parameters()):
            t_params.data = s_params.data


    trainSteps = int(((len(train_dataset_unlabel) + len(train_dataset)) / config.TRAIN.BATCH_SIZE_PER_GPU / len(config.GPUS)) * config.TRAIN.END_EPOCH)
    validPerStep = int(((len(train_dataset_unlabel) + len(train_dataset)) / config.TRAIN.BATCH_SIZE_PER_GPU / len(config.GPUS)) * config.TRAIN.VALID_PER_EPOCH)
    beginSTStep = int(((len(train_dataset_unlabel) + len(train_dataset)) / config.TRAIN.BATCH_SIZE_PER_GPU / len(config.GPUS)) * config.TRAIN.ST_BEGIN_EPOCH)
    againSTStep = int(((len(train_dataset_unlabel) + len(train_dataset)) / config.TRAIN.BATCH_SIZE_PER_GPU / len(config.GPUS)) * config.TRAIN.ST_AGAIN_EPOCH)
    logger.info('Train Steps:{}, valid Per Step:{}, self-training begin step:{}, self-training again step:{}'.format(
        trainSteps, validPerStep, beginSTStep, againSTStep))
    best_mIoU = 0
    best_PA = 0
    start_unsuper = 150
    epoch = config.TRAIN.END_EPOCH



    for i_epoch in range(1, epoch+1):


        if i_epoch <= start_unsuper:

            print('supervised', i_epoch)
            for iter, batch in tqdm(enumerate(trainloader_gt)):
                i_iter = i_epoch * (len(train_dataset) / batch_size) + iter
                lambdaST = adjustLambdaST(i_iter, beginSTStep, againSTStep, config.TRAIN.ST_AF)
                optimizer.zero_grad()
                lr = adjust_learning_rate(optimizer, i_iter, trainSteps, config)
                model.train()
                model_teacher.eval()
                images, labels, _, _, _ = batch
                images = Variable(images).cuda()
                pred_list = model(images)
                pred_list_ce = pred_list[:2]
                if config.LOSS.USE_OHEM:
                    loss_ce = loss_calc_ohem(pred_list_ce, labels, config)
                    loss_ce = loss_ce.mean()
                else:
                    loss_ce = loss_calc(pred_list_ce, labels, config)
                loss_sup = loss_ce
                loss_sup.backward()
                optimizer.step()

                with torch.no_grad():
                    # epoch = i_iter / validPerStep
                    if i_epoch > start_unsuper:  # cfg["trainer"].get("sup_only_epoch", 0):
                        ema_decay = min(1 - 1 / (i_iter - len(train_dataset) * 0 + 1),
                                        0.999,
                                        )
                    else:
                        ema_decay = 0.
                        # ema_decay = 0.99
                    # update weight
                    for param_train, param_eval in zip(model.parameters(), model_teacher.parameters()):
                        param_eval.data = param_eval.data * ema_decay + param_train.data * (1 - ema_decay)
                    # update bn
                    for buffer_train, buffer_eval in zip(model.buffers(), model_teacher.buffers()):
                        buffer_eval.data = buffer_eval.data * ema_decay + buffer_train.data * (1 - ema_decay)
                        # buffer_eval.data = buffer_train.data

                del batch, images, labels, pred_list_ce, pred_list


        else:
            print('semi-supervised', i_epoch)

            iter = 0
            for bar_supervised, bar_unsupervised in zip(trainloader_gt, trainloader_unlabel):
                iter += 1
                i_iter = i_epoch * (len(train_dataset) / batch_size) + iter
                lambdaST = adjustLambdaST(i_iter, beginSTStep, againSTStep, config.TRAIN.ST_AF)
                optimizer.zero_grad()
                lr = adjust_learning_rate(optimizer, i_iter, trainSteps, config)


                model.train()
                model_teacher.eval()
                images, labels, _, _, _ = bar_supervised
                images = Variable(images).cuda()


                images_weak, images_aug1, images_aug2, cutmix1, cutmix2 = bar_unsupervised

                images_weak, images_aug1, images_aug2, cutmix1, cutmix2 = Variable(images_weak).cuda(), Variable(images_aug1).cuda(), Variable(images_aug2).cuda(), cutmix1.cuda(), cutmix2.cuda()

                with torch.no_grad():
                    pred_list_T = model_teacher(images_weak)
                    pesudo_label = pred_list_T[1]
                    pesudo_label = pesudo_label.softmax(dim=1)
                    pesudo_label = torch.argmax(pesudo_label, dim=1)


                num_lb, num_ulb = images.shape[0], images_aug1.shape[0]
                aux, pred, fea_con, pred_con = model(torch.cat((images, images_aug1)))
                pred_list_lb = [aux[:num_lb], pred[:num_lb], fea_con[:num_lb], pred_con[:num_lb]]
                pred_list_ulb = [aux[num_lb : ], pred[num_lb : ], fea_con[num_lb : ], pred_con[num_lb : ]]

                if config.LOSS.USE_OHEM:
                    loss_ce = loss_calc_ohem(pred_list_lb[:2], labels, config)
                    loss_ce = loss_ce.mean()
                else:
                    loss_ce = loss_calc(pred_list_lb[:2], labels, config)
                loss_sup = loss_ce

                if config.LOSS.USE_OHEM:
                    loss_unsup_weak = loss_calc_ohem(pred_list_ulb[:2], pesudo_label, config)
                    loss_unsup_weak = loss_unsup_weak.mean()
                else:
                    loss_unsup_weak = loss_calc(pred_list_ulb[:2], pesudo_label, config)
                loss_unsup_weak = loss_unsup_weak
                loss = (loss_sup + loss_unsup_weak) / 2.0

                loss.backward()
                optimizer.step()

        # 更新teacher
        # update teacher model with EMA

                with torch.no_grad():
                    # epoch = i_iter / validPerStep
                    if i_epoch > start_unsuper:#cfg["trainer"].get("sup_only_epoch", 0):
                        ema_decay = min(1 - 1 / ( i_iter - len(train_dataset ) * 0 + 1 ),
                            0.999,
                        )
                    else:
                        ema_decay = 0.
                        # ema_decay = 0.99
                    # update weight
                    for param_train, param_eval in zip(model.parameters(), model_teacher.parameters()):
                        param_eval.data = param_eval.data * ema_decay + param_train.data * (1 - ema_decay)
                    # update bn
                    for buffer_train, buffer_eval in zip(model.buffers(), model_teacher.buffers()):
                        buffer_eval.data = buffer_eval.data * ema_decay + buffer_train.data * (1 - ema_decay)
                        # buffer_eval.data = buffer_train.data

        # test
        logger.info('start valid')
        mean_IoU, IoU_array, PA, f1 = validate(config, testloader, model)
        logger.info('=> saving checkpoint to {}'.format(os.path.join(rootOutputDir, 'checkpoint_S.pth')))
        logger.info('=> saving checkpoint_D to {}'.format(os.path.join(rootOutputDir, 'checkpoint_T.pth')))
        torch.save(model.module.state_dict(), os.path.join(rootOutputDir, 'checkpoint_S.pth'))
        torch.save(model_teacher.module.state_dict(), os.path.join(rootOutputDir, 'checkpoint_T.pth'))
        if mean_IoU > best_mIoU:
            best_mIoU = mean_IoU
            torch.save({
                'optimizer': optimizer.state_dict(),
                'model': model.module.state_dict(),
                'model_teacher': model_teacher.module.state_dict(),
                # 'RC_memory': RC_memory.state_dict(),
            }, os.path.join(rootOutputDir, 'best_mIoU.pth'))
            logger.info('=> saving best_mIoU to {}'.format(os.path.join(rootOutputDir, 'best_mIoU.pth')))
        if PA > best_PA:
            best_PA = PA
            torch.save(model.module.state_dict(), os.path.join(rootOutputDir, 'best_PA.pth'))
            logger.info('=> saving best_PA to {}'.format(os.path.join(rootOutputDir, 'best_PA.pth')))
        msg = 'epoch: {}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}, PA: {: 4.4f}, Best_PA: {: 4.4f}, f1: {: 4.4f}'.format(
            i_epoch, mean_IoU, best_mIoU, PA, best_PA, f1)
        logger.info(msg)
        logger.info('iou_class:{: 4.4f}, {: 4.4f}, {: 4.4f}, {: 4.4f}, {: 4.4f}, {: 4.4f} '.format(IoU_array[0],
                                                                                                   IoU_array[1],
                                                                                                   IoU_array[2],
                                                                                                   IoU_array[3],
                                                                                                   IoU_array[4],
                                                                                                   IoU_array[5]))
        model.train()

        # use best model
        if config.TRAIN.USE_BEST_MODEL and lambdaST != 0.0:
            best_mIoU_file = os.path.join(rootOutputDir, 'best_mIoU.pth')
            if os.path.isfile(best_mIoU_file):
                checkpoint_mIoU = torch.load(best_mIoU_file)
                optimizer.load_state_dict(checkpoint_mIoU['optimizer'])
                model.module.load_state_dict(checkpoint_mIoU['model'])
                model_teacher.module.load_state_dict(checkpoint_mIoU['model_teacher'])
                # RC_memory.load_state_dict(checkpoint_mIoU['RC_memory'])
                logger.info('=> loaded best_mIoU_model')





if __name__ == '__main__':

    main()



