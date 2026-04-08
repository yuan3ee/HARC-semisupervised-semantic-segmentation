#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelzoo

# import our stuffs
from .bn_helper import BatchNorm2d, BatchNorm2d_class, relu_inplace
from .resnet import ResNetBackbone


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


class Seg_Resnet50_unet(nn.Module):
    def __init__(self, cfg):
        super(Seg_Resnet50_unet, self).__init__()
        self.nclass = cfg.DATASET.NUM_CLASSES
        self.pretrained = ResNetBackbone(cfg)(arch = 'resnet50')
        self.score1=nn.Sequential(nn.Conv2d(512, self.nclass, 1))
        self.score2=nn.Sequential(nn.Conv2d(1024, self.nclass, 1))
        self.score3 =_FCNHead(2048, self.nclass, BatchNorm2d)
        self.fea_contrast = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.pred_contrast = nn.Sequential(nn.Conv2d(256, self.nclass, 1))
        self.aux_head = nn.Conv2d(1024, self.nclass, 1)

    def base_forward(self, x):
        x, feat8, feat16, feat32 = self.pretrained(x)
        return x, feat8, feat16, feat32

    def forward(self, x):
        _, _, h, w = x.size()
        c1, c2, c3, c4 = self.base_forward(x)
        aux_fuse = self.aux_head(c3)

        c4_fea = self.fea_contrast(c4)
        c4_pred = self.pred_contrast(c4_fea)

        c4_score = self.score3(c4)
        c3_score = self.score2(c3)
        c2_score = self.score1(c2)
        upscore3 = F.interpolate(c4_score, c3_score.size()[2:], mode='bilinear', align_corners=True)
        fuse = upscore3 + c3_score
        fuse = F.interpolate(fuse, c2_score.size()[2:], mode='bilinear', align_corners=True)
        fuse = fuse + c2_score
        fuse = F.interpolate(fuse,(h,w), mode='bilinear', align_corners=True)
        return aux_fuse, fuse, c4_fea, c4_pred

class region_memnory(nn.Module):
    def __init__(self, cfg, inner_planes=256):
        super(region_memnory, self).__init__()
        self.nclass = cfg.DATASET.NUM_CLASSES
        if cfg.DATASET.DATASET == 'postdam':
            if cfg.DATASET.LABELED_RATIO == 0.125: # self.queue_len是用于对比学习数据集的大小
                self.queue_len = 5376
            elif cfg.DATASET.LABELED_RATIO == 0.25:
                self.queue_len = 4608
        elif cfg.DATASET.DATASET == 'vaihingen':
            if cfg.DATASET.LABELED_RATIO == 0.125:  # self.queue_len是用于对比学习数据集的大小
                self.queue_len = 492
            elif cfg.DATASET.LABELED_RATIO == 0.25:
                self.queue_len = 423
        elif cfg.DATASET.DATASET == 'loveda':
            if cfg.DATASET.LABELED_RATIO == 0.125:  # self.queue_len是用于对比学习数据集的大小
                self.queue_len = 8827
            elif cfg.DATASET.LABELED_RATIO == 0.25:
                self.queue_len = 7566
        self.queue = torch.randn(size=[self.nclass, inner_planes, self.queue_len])
        self.ptr = torch.zeros(1,dtype=torch.long)
        for i in range(self.nclass):
            self.register_buffer("queue"+str(i),torch.randn(inner_planes, self.queue_len))
            self.register_buffer("ptr"+str(i),torch.zeros(1,dtype=torch.long))
            exec("self.queue"+str(i) + '=' + 'nn.functional.normalize(' + "self.queue"+str(i) + ',dim=0)')

    def _dequeue_and_enqueue(self,keys,vals,cat, bs):
        if cat not in vals:
            return
        keys = keys[list(vals).index(cat)]
        batch_size = bs
        ptr = int(eval("self.ptr"+str(cat)))
        # print('ptr', ptr.device)
        eval("self.queue"+str(cat))[:,ptr] = keys
        # print('queue'+str(cat)+':', eval("self.queue"+str(cat))[:,ptr].shape,keys.shape)
        ptr = (ptr + batch_size) % self.queue_len
        eval("self.ptr"+str(cat))[0] = ptr
        # print('cat', cat)


    def construct_region(self, fea, pred):   # fea:[4, 256, 65, 65]  pred[4, num_class, 65, 65]
        bs = fea.shape[0]
        pred = pred.max(1)[1].squeeze().view(bs, -1) # pred.shape = [4, 65*65=4225]
        val = torch.unique(pred) # 挑出pred独立的不重复的元素，即这张图中存在的种类 val=[0,1,2,3,4,5]
        fea=fea.squeeze()
        fea = fea.view(bs, 256,-1).permute(1,0,2) # fea.shape=[256, 4, 4225]
        new_fea = fea[:,pred==val[0]].mean(1).unsqueeze(0) # new_fea.shape=[1, 256], 此处为0类别的region center
        for i in val[1:]:
            if(i<self.nclass):
                class_fea = fea[:,pred==i].mean(1).unsqueeze(0)
                new_fea = torch.cat((new_fea,class_fea),dim=0)
        val = torch.tensor([i for i in val if i<self.nclass])
        return new_fea, val #.cuda() # new_fea.shape = [6, 256], val=[0,1,2,3,4,5]

    def forward(self, fea, res, batch_size=8, mode_for='supervised'):
        bs = batch_size
        keys, vals = self.construct_region(fea, res)
        keys = nn.functional.normalize(keys, dim=1)
        if mode_for == 'supervised':
            for i in range(self.nclass):
                self._dequeue_and_enqueue(keys,vals,i, bs)
        return  keys, vals


def region_memory(num_class):
    region_dic = {}
    for i in range(num_class):
        region_dic['region_class'+str(i)] = torch.zeros(size=[256,1])
        # region_dic['region_class'+str(i)] = torch.randn(size=[256,1],)
    return region_dic

def computer_region_center(fea, pred, num_class):
    bs = fea.shape[0]
    pred = pred.max(1)[1].squeeze().view(bs, -1)  # pred.shape = [4, 65*65=4225]
    val = torch.unique(pred)  # 挑出pred独立的不重复的元素，即这张图中存在的种类 val=[0,1,2,3,4,5]
    fea = fea.squeeze()
    fea = fea.view(bs, 256, -1).permute(1, 0, 2)  # fea.shape=[256, 4, 4225]
    new_fea = fea[:, pred == val[0]].mean(1).unsqueeze(0)  # new_fea.shape=[1, 256], 此处为0类别的region center
    for i in val[1:]:
        if (i < num_class):
            class_fea = fea[:, pred == i].mean(1).unsqueeze(0)
            new_fea = torch.cat((new_fea, class_fea), dim=0)
    val = torch.tensor([i for i in val if i < num_class])
    new_fea = nn.functional.normalize(new_fea, dim=1)
    return new_fea, val  # .cuda() # new_fea.shape = [6, 256], v


def update_region_memory(region_dic, keys, vals, cat):
    if cat not in vals:
        return region_dic
    keys = keys[list(vals).index(cat)]
    keys = torch.unsqueeze(keys, dim=-1)
    # print(region_dic['region_class'+str(cat)].shape)
    # if region_dic['region_class'+str(cat)].shape[-1] == 1:
    # # if region_dic['region_class' + str(cat)].bool().all() == 0:
    #     keys = torch.unsqueeze(keys,dim=-1)
    #     region_dic['region_class'+str(cat)] = keys
    # else:
    #     print('11111')
    #     print(region_dic['region_class'+str(cat)].shape)
    #     print('keys', keys.shape)
    #     keys = torch.unsqueeze(keys, dim=-1)
    #     region_dic['region_class' + str(cat)] = torch.cat((region_dic['region_class'+str(cat)], keys), dim=-1)
    # print('region_class'+str(cat), region_dic['region_class'+str(cat)], region_dic['region_class'+str(cat)].shape)
    # print('keys', keys.shape)
    # region_dic['region_class' + str(cat)].cuda().cat(keys, dim=-1)
    try:
    #     region_dic['region_class' + str(cat)].cuda().cat(keys, dim=-1)
        region_dic['region_class' + str(cat)] = torch.cat((region_dic['region_class' + str(cat)].cuda(), keys), dim=-1)
    except:
        return region_dic
    return region_dic


criterion = nn.CrossEntropyLoss()

def compute_contrast_loss(l_pos, l_neg, temperature=0.2):
    N = l_pos.size(0)
    logits = torch.cat((l_pos, l_neg), dim=1)
    logits /= temperature
    labels = torch.zeros((N,), dtype=torch.long).cuda()
    return criterion(logits, labels)

def region_contrast_loss(cfg, keys, vals, region_memory):
    contrast_loss = 0
    for cls_ind in range(cfg.DATASET.NUM_CLASSES):
        if cls_ind in vals:
            # 是否会遇到query的元素都是0，即当前这批次图片没有任何像素被预测为该类别
            query = keys[list(vals).index(cls_ind)]  # 256,
            l_pos = query.unsqueeze(1) * region_memory['region_class' + str(cls_ind)].clone().detach()
            # try:
            #     l_pos = query.unsqueeze(1) * region_memory['region_class'+str(cls_ind)].clone().detach()
            # except:
            #     print('cls_ind', cls_ind)
            #     if region_memory['region_class'+str(cls_ind)] is None:
            #         print('region_memory'+str(cls_ind))
            #         print(region_memory['region_class'+str(cls_ind)])
            #     break

            # l_pos = query.unsqueeze(1) * eval("region_memory.queue" + str(cls_ind)).clone().detach()  # 256, N1
            all_ind = [m for m in range(cfg.DATASET.NUM_CLASSES)]
            l_neg = 0
            tmp = all_ind.copy()
            # print('cls_ind',cls_ind)
            # print('tmp', tmp)
            tmp.remove(cls_ind)
            for cls_ind2 in tmp:
                try:
                    l_neg += query.unsqueeze(1) * region_memory['region_class'+str(cls_ind2)].clone().detach()
                except:
                    new = query.unsqueeze(1) * region_memory['region_class' + str(cls_ind2)].clone().detach()
                    # print('new', new.shape)
                    # print('l_neg', l_neg.shape)
                    # print('l_pos', l_pos.shape)
                    # print('cls', cls_ind)
                    # print('cls2', cls_ind2)
                    if new.shape[-1] < l_neg.shape[-1]:
                        dims = l_neg.shape[-1] - new.shape[-1]
                        new = torch.cat((new, torch.zeros(size=[256, dims]).cuda()), dim=-1)
                        l_neg += new
                    elif new.shape[-1] > l_neg.shape[-1]:
                        dims = new.shape[-1] - l_neg.shape[-1]
                        l_neg = torch.cat((l_neg, torch.zeros(size=[256, dims]).cuda()), dim=-1)
                        l_neg += new
                    # print('query.unsqueeze(1)', query.unsqueeze(1).shape)
                    # print('region'+str(cls_ind2), region_memory['region_class'+str(cls_ind2)].shape)
                    # print('l_neg', l_neg.shape)
                    # print('l_pos', l_pos.shape)
                # l_neg += query.unsqueeze(1) * eval("region_memory.queue" + str(cls_ind2)).clone().detach()
            contrast_loss += compute_contrast_loss(l_pos, l_neg)
        else:
            continue
    return contrast_loss

def get_seg_model(cfg, **kwargs):
    model = Seg_Resnet50_unet(cfg)
    return model