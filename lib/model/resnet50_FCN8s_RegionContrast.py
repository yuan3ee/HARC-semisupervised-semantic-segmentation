#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelzoo
from skimage import measure
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
        self.score3 =_FCNHead(2048, 256, BatchNorm2d)
        self.pred_layer = nn.Sequential(nn.Conv2d(256, self.nclass, 1))
        self.aux_head = nn.Conv2d(1024, self.nclass, 1)

    def base_forward(self, x):
        x, feat8, feat16, feat32 = self.pretrained(x)
        return x, feat8, feat16, feat32

    def forward(self, x):
        _, _, h, w = x.size()
        c1, c2, c3, c4 = self.base_forward(x)
        aux_fuse = self.aux_head(c3)
        c4_fea = self.score3(c4)#最底层特征 fea
        c4_score = self.pred_layer(c4_fea)
        c3_score = self.score2(c3)
        c2_score = self.score1(c2)
        upscore3 = F.interpolate(c4_score, c3_score.size()[2:], mode='bilinear', align_corners=True)
        fuse = upscore3 + c3_score
        fuse = F.interpolate(fuse, c2_score.size()[2:], mode='bilinear', align_corners=True)
        fuse = fuse + c2_score
        fuse = F.interpolate(fuse,(h,w), mode='bilinear', align_corners=True)

        return aux_fuse, fuse, c4_fea, c4_score
        # fuse
        # torch.Size([8, 6, 512, 512])
        # aux_fuse
        # torch.Size([8, 6, 33, 33])
        # c4_fea
        # torch.Size([8, 256, 17, 17])
        # c4_score
        # torch.Size([8, 6, 17, 17])

class RegionContrast(nn.Module):
    def __init__(self, cfg, in_planes, num_class, temperature=0.2):
        super(RegionContrast, self).__init__()
        self.in_planes = in_planes
        self.num_classes = num_class
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        if cfg.DATASET.LABELED_RATIO is not None:  # self.queue_len是用于对比学习数据集的大小
            self.queue_len = int(cfg.DATASET.LABELED_RATIO * cfg.DATASET.TRAINSIZE)
        else:
            self.queue_len = cfg.DATASET.TRAINSIZE
        for i in range(self.num_classes):
            self.register_buffer("queue"+str(i),torch.randn(self.in_planes, self.queue_len))
            self.register_buffer("ptr"+str(i),torch.zeros(1,dtype=torch.long))
            exec("self.queue"+str(i) + '=' + 'nn.functional.normalize(' + "self.queue"+str(i) + ',dim=0)')

    def forward(self, fea, pred):
        bs = fea.shape[0]
        keys, vals = self.construct_region(fea, pred)
        keys = nn.functional.normalize(keys, dim=1)  # 在指定维度下计算2范数
        contrast_loss = 0
        contras_loss_class_0, n_0 = 0, 0
        contras_loss_class_1, n_1 = 0, 0
        contras_loss_class_2, n_2 = 0, 0
        contras_loss_class_3, n_3 = 0, 0
        contras_loss_class_4, n_4 = 0, 0
        contras_loss_class_5, n_5 = 0, 0
        for cls_ind in range(self.num_classes):
            if cls_ind in vals:
                # 是否会遇到query的元素都是0，即当前这批次图片没有任何像素被预测为该类别
                query = keys[list(vals).index(cls_ind)]   #256,
                l_pos = query.unsqueeze(1)*eval("self.queue"+str(cls_ind)).clone().detach()  #256, N1
                all_ind = [m for m in range(6)]
                l_neg = 0
                tmp = all_ind.copy()
                tmp.remove(cls_ind)
                for cls_ind2 in tmp:
                    l_neg += query.unsqueeze(1)*eval("self.queue"+str(cls_ind2)).clone().detach()
                # if cls_ind == 0:
                #     contras_loss_class_0 = self._compute_contrast_loss(l_pos, l_neg)
                # elif cls_ind == 1:
                #     contras_loss_class_1 = self._compute_contrast_loss(l_pos, l_neg)
                # elif cls_ind == 2:
                #     contras_loss_class_2 = self._compute_contrast_loss(l_pos, l_neg)
                # elif cls_ind == 3:
                #     contras_loss_class_3 = self._compute_contrast_loss(l_pos, l_neg)
                # elif cls_ind == 4:
                #     contras_loss_class_4 = self._compute_contrast_loss(l_pos, l_neg)
                # elif cls_ind == 5:
                #     contras_loss_class_5 = self._compute_contrast_loss(l_pos, l_neg)
                contrast_loss += self._compute_contrast_loss(l_pos, l_neg)
            else:
                continue
        for i in range(self.num_classes):
            self._dequeue_and_enqueue(keys,vals,i, bs)
        # return contras_loss_class_0, contras_loss_class_1,contras_loss_class_2,contras_loss_class_3,contras_loss_class_4,contras_loss_class_5
        return contrast_loss

    def construct_region(self, fea, pred):   # fea:[4, 256, 65, 65]  pred[4, num_class, 65, 65]
        bs = fea.shape[0]
        pred = pred.max(1)[1].squeeze().view(bs, -1) # pred.shape = [4, 65*65=4225]
        val = torch.unique(pred) # 挑出pred独立的不重复的元素，即这张图中存在的种类 val=[0,1,2,3,4,5]
        fea=fea.squeeze()
        fea = fea.view(bs, 256,-1).permute(1,0,2) # fea.shape=[256, 4, 4225]
        new_fea = fea[:,pred==val[0]].mean(1).unsqueeze(0) # new_fea.shape=[1, 256], 此处为0类别的region center
        for i in val[1:]:
            if(i<6):
                class_fea = fea[:,pred==i].mean(1).unsqueeze(0)
                new_fea = torch.cat((new_fea,class_fea),dim=0)
        val = torch.tensor([i for i in val if i<6])
        return new_fea, val.cuda() # new_fea.shape = [6, 256], val=[0,1,2,3,4,5]

    def _compute_contrast_loss(self, l_pos, l_neg):
        N = l_pos.size(0)
        logits = torch.cat((l_pos,l_neg),dim=1)
        logits /= self.temperature
        labels = torch.zeros((N,),dtype=torch.long).cuda()
        return self.criterion(logits,labels)

    def _dequeue_and_enqueue(self,keys,vals,cat, bs):
        if cat not in vals:
            return
        keys = keys[list(vals).index(cat)]
        batch_size = bs
        ptr = int(eval("self.ptr"+str(cat)))
        eval("self.queue"+str(cat))[:,ptr] = keys
        ptr = (ptr + batch_size) % self.queue_len
        eval("self.ptr"+str(cat))[0] = ptr

# Discriminator 的功能是计算出对比损失进行判断，返回可用的伪标签类别
class RegionContrast_Discriminator(nn.Module):
    def __init__(self, cfg, in_planes, num_class, temperature=0.2):
        super(RegionContrast_Discriminator, self).__init__()
        self.in_planes = in_planes
        self.num_classes = num_class
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        if cfg.DATASET.LABELED_RATIO is not None:  # self.queue_len是用于对比学习数据集的大小
            self.queue_len = int(cfg.DATASET.LABELED_RATIO * cfg.DATASET.TRAINSIZE)
        else:
            self.queue_len = cfg.DATASET.TRAINSIZE
        for i in range(self.num_classes):
            self.register_buffer("queue"+str(i),torch.randn(self.in_planes, self.queue_len))
            self.register_buffer("ptr"+str(i),torch.zeros(1,dtype=torch.long))
            exec("self.queue"+str(i) + '=' + 'nn.functional.normalize(' + "self.queue"+str(i) + ',dim=0)')

    def forward(self, fea, pred, contrast_loss_input, pesudo_label):
        pesudo_label = torch.argmax(pesudo_label, dim=1).float()
        contrast_loss_input = torch.tensor(contrast_loss_input)
        bs = fea.shape[0]
        keys, vals = self.construct_region(fea, pred)
        keys = nn.functional.normalize(keys, dim=1)  # 在指定维度下计算2范数
        contrast_loss = 0
        for cls_ind in range(self.num_classes):
            if cls_ind in vals:
                # 是否会遇到query的元素都是0，即当前这批次图片没有任何像素被预测为该类别
                query = keys[list(vals).index(cls_ind)]   #256,
                l_pos = query.unsqueeze(1)*eval("self.queue"+str(cls_ind)).clone().detach()  #256, N1
                all_ind = [m for m in range(6)]
                l_neg = 0
                tmp = all_ind.copy()
                tmp.remove(cls_ind)
                for cls_ind2 in tmp:
                    l_neg += query.unsqueeze(1)*eval("self.queue"+str(cls_ind2)).clone().detach()
                contrast_loss_class = self._compute_contrast_loss(l_pos, l_neg)
                contrast_loss += contrast_loss_class
                if contrast_loss_class > contrast_loss_input[cls_ind]:
                    pesudo_label = torch.where(pesudo_label==cls_ind, torch.tensor(-1).float().cuda(), pesudo_label)
            else:
                pesudo_label = torch.where(pesudo_label == cls_ind, torch.tensor(-1).float().cuda(), pesudo_label)
        return pesudo_label

    def construct_region(self, fea, pred):   # fea:[4, 256, 65, 65]  pred[4, num_class, 65, 65]
        bs = fea.shape[0]
        pred = pred.max(1)[1].squeeze().view(bs, -1) # pred.shape = [4, 65*65=4225]
        val = torch.unique(pred) # 挑出pred独立的不重复的元素，即这张图中存在的种类 val=[0,1,2,3,4,5]
        fea=fea.squeeze()
        fea = fea.view(bs, 256,-1).permute(1,0,2) # fea.shape=[256, 4, 4225]
        new_fea = fea[:,pred==val[0]].mean(1).unsqueeze(0) # new_fea.shape=[1, 256], 此处为0类别的region center
        for i in val[1:]:
            if(i<6):
                class_fea = fea[:,pred==i].mean(1).unsqueeze(0)
                new_fea = torch.cat((new_fea,class_fea),dim=0)
        val = torch.tensor([i for i in val if i<6])
        return new_fea, val.cuda() # new_fea.shape = [6, 256], val=[0,1,2,3,4,5]

    def _compute_contrast_loss(self, l_pos, l_neg):
        N = l_pos.size(0)
        logits = torch.cat((l_pos,l_neg),dim=1)
        logits /= self.temperature
        labels = torch.zeros((N,),dtype=torch.long).cuda()
        return self.criterion(logits,labels)

    def _dequeue_and_enqueue(self,keys,vals,cat, bs):
        if cat not in vals:
            return
        keys = keys[list(vals).index(cat)]
        batch_size = bs
        ptr = int(eval("self.ptr"+str(cat)))
        eval("self.queue"+str(cat))[:,ptr] = keys
        ptr = (ptr + batch_size) % self.queue_len
        eval("self.ptr"+str(cat))[0] = ptr

class RegionContrast_all(nn.Module):
    def __init__(self, cfg, in_planes, num_class, temperature=0.2):
        super(RegionContrast, self).__init__()
        self.in_planes = in_planes
        self.num_classes = num_class
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        if cfg.DATASET.LABELED_RATIO is not None:  # self.queue_len是用于对比学习数据集的大小
            self.queue_len = int(cfg.DATASET.LABELED_RATIO * cfg.DATASET.TRAINSIZE)
        else:
            self.queue_len = cfg.DATASET.TRAINSIZE
        for i in range(self.num_classes):
            self.register_buffer("queue"+str(i),torch.randn(self.in_planes, self.queue_len))
            self.register_buffer("ptr"+str(i),torch.zeros(1,dtype=torch.long))
            exec("self.queue"+str(i) + '=' + 'nn.functional.normalize(' + "self.queue"+str(i) + ',dim=0)')

    def forward(self, fea, pred, contrast_loss_input=None, pesudo_lable=None, stage='supervised'):
        if stage == 'supervised':
            bs = fea.shape[0]
            keys, vals = self.construct_region(fea, pred)
            keys = nn.functional.normalize(keys, dim=1)  # 在指定维度下计算2范数
            contrast_loss = 0
            for cls_ind in range(self.num_classes):
                if cls_ind in vals:
                    # 是否会遇到query的元素都是0，即当前这批次图片没有任何像素被预测为该类别
                    query = keys[list(vals).index(cls_ind)]  # 256,
                    l_pos = query.unsqueeze(1) * eval("self.queue" + str(cls_ind)).clone().detach()  # 256, N1
                    all_ind = [m for m in range(6)]
                    l_neg = 0
                    tmp = all_ind.copy()
                    tmp.remove(cls_ind)
                    for cls_ind2 in tmp:
                        l_neg += query.unsqueeze(1) * eval("self.queue" + str(cls_ind2)).clone().detach()
                    contrast_loss += self._compute_contrast_loss(l_pos, l_neg)
                else:
                    continue
            for i in range(self.num_classes):
                self._dequeue_and_enqueue(keys, vals, i, bs)
            return contrast_loss
        else:
            pesudo_lable = torch.argmax(pesudo_lable, dim=1)
            bs = fea.shape[0]
            keys, vals = self.construct_region(fea, pred)
            keys = nn.functional.normalize(keys, dim=1)  # 在指定维度下计算2范数
            contrast_loss = 0
            for cls_ind in range(self.num_classes):
                if cls_ind in vals:
                    # 是否会遇到query的元素都是0，即当前这批次图片没有任何像素被预测为该类别
                    query = keys[list(vals).index(cls_ind)]   #256,
                    l_pos = query.unsqueeze(1)*eval("self.queue"+str(cls_ind)).clone().detach()  #256, N1
                    all_ind = [m for m in range(6)]
                    l_neg = 0
                    tmp = all_ind.copy()
                    tmp.remove(cls_ind)
                    for cls_ind2 in tmp:
                        l_neg += query.unsqueeze(1)*eval("self.queue"+str(cls_ind2)).clone().detach()
                    contrast_loss_class = self._compute_contrast_loss(l_pos, l_neg)
                    contrast_loss += contrast_loss_class
                    if contrast_loss_class > contrast_loss_input[cls_ind]:
                         torch.where(pesudo_lable, cls_ind, -1)
                else:
                    torch.where(pesudo_lable, cls_ind, -1)
            return pesudo_lable

    def construct_region(self, fea, pred):   # fea:[4, 256, 65, 65]  pred[4, num_class, 65, 65]
        bs = fea.shape[0]
        pred = pred.max(1)[1].squeeze().view(bs, -1) # pred.shape = [4, 65*65=4225]
        val = torch.unique(pred) # 挑出pred独立的不重复的元素，即这张图中存在的种类 val=[0,1,2,3,4,5]
        fea=fea.squeeze()
        fea = fea.view(bs, 256,-1).permute(1,0,2) # fea.shape=[256, 4, 4225]
        new_fea = fea[:,pred==val[0]].mean(1).unsqueeze(0) # new_fea.shape=[1, 256], 此处为0类别的region center
        for i in val[1:]:
            if(i<6):
                class_fea = fea[:,pred==i].mean(1).unsqueeze(0)
                new_fea = torch.cat((new_fea,class_fea),dim=0)
        val = torch.tensor([i for i in val if i<6])
        return new_fea, val.cuda() # new_fea.shape = [6, 256], val=[0,1,2,3,4,5]

    def _compute_contrast_loss(self, l_pos, l_neg):
        N = l_pos.size(0)
        logits = torch.cat((l_pos,l_neg),dim=1)
        logits /= self.temperature
        labels = torch.zeros((N,),dtype=torch.long).cuda()
        return self.criterion(logits,labels)

    def _dequeue_and_enqueue(self,keys,vals,cat, bs):
        if cat not in vals:
            return
        keys = keys[list(vals).index(cat)]
        batch_size = bs
        ptr = int(eval("self.ptr"+str(cat)))
        eval("self.queue"+str(cat))[:,ptr] = keys
        ptr = (ptr + batch_size) % self.queue_len
        eval("self.ptr"+str(cat))[0] = ptr


# Discriminator 的功能是计算出对比损失进行判断，返回可用的伪标签类别,此为连同域判断版本
class RegionContrast_Discriminator_liantongyu(nn.Module):
    def __init__(self, cfg, in_planes, num_class, temperature=0.2):
        super(RegionContrast_Discriminator, self).__init__()
        self.in_planes = in_planes
        self.num_classes = num_class
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        if cfg.DATASET.LABELED_RATIO is not None:  # self.queue_len是用于对比学习数据集的大小
            self.queue_len = int(cfg.DATASET.LABELED_RATIO * cfg.DATASET.TRAINSIZE)
        else:
            self.queue_len = cfg.DATASET.TRAINSIZE
        for i in range(self.num_classes):
            self.register_buffer("queue"+str(i),torch.randn(self.in_planes, self.queue_len))
            self.register_buffer("ptr"+str(i),torch.zeros(1,dtype=torch.long))
            exec("self.queue"+str(i) + '=' + 'nn.functional.normalize(' + "self.queue"+str(i) + ',dim=0)')

    def forward(self, fea, pred, contrast_loss_input, pesudo_label):
        pesudo_label = torch.argmax(pesudo_label, dim=1).float()
        contrast_loss_input = torch.tensor(contrast_loss_input)
        bs = fea.shape[0]
        keys, vals = self.construct_region(fea, pred)
        keys = nn.functional.normalize(keys, dim=1)  # 在指定维度下计算2范数
        contrast_loss = 0
        for cls_ind in range(self.num_classes):
            if cls_ind in vals:
                # 是否会遇到query的元素都是0，即当前这批次图片没有任何像素被预测为该类别
                query = keys[list(vals).index(cls_ind)]   #256,
                l_pos = query.unsqueeze(1)*eval("self.queue"+str(cls_ind)).clone().detach()  #256, N1
                all_ind = [m for m in range(6)]
                l_neg = 0
                tmp = all_ind.copy()
                tmp.remove(cls_ind)
                for cls_ind2 in tmp:
                    l_neg += query.unsqueeze(1)*eval("self.queue"+str(cls_ind2)).clone().detach()
                contrast_loss_class = self._compute_contrast_loss(l_pos, l_neg)
                contrast_loss += contrast_loss_class
                if contrast_loss_class > contrast_loss_input[cls_ind]:
                    pesudo_label = torch.where(pesudo_label==cls_ind, torch.tensor(-1).float().cuda(), pesudo_label)
            else:
                pesudo_label = torch.where(pesudo_label == cls_ind, torch.tensor(-1).float().cuda(), pesudo_label)
        return pesudo_label

    def construct_region(self, fea, pred):   # fea:[4, 256, 65, 65]  pred[4, num_class, 65, 65]
        bs = fea.shape[0]
        pred = pred.max(1)[1].squeeze().view(bs, -1) # pred.shape = [4, 65*65=4225]
        val = torch.unique(pred) # 挑出pred独立的不重复的元素，即这张图中存在的种类 val=[0,1,2,3,4,5]
        fea=fea.squeeze()
        fea = fea.view(bs, 256,-1).permute(1,0,2) # fea.shape=[256, 4, 4225]
        new_fea = fea[:,pred==val[0]].mean(1).unsqueeze(0) # new_fea.shape=[1, 256], 此处为0类别的region center
        for i in val[1:]:
            if(i<6):
                class_fea = fea[:,pred==i].mean(1).unsqueeze(0)
                new_fea = torch.cat((new_fea,class_fea),dim=0)
        val = torch.tensor([i for i in val if i<6])
        return new_fea, val.cuda() # new_fea.shape = [6, 256], val=[0,1,2,3,4,5]

    def _compute_contrast_loss(self, l_pos, l_neg):
        N = l_pos.size(0)
        logits = torch.cat((l_pos,l_neg),dim=1)
        logits /= self.temperature
        labels = torch.zeros((N,),dtype=torch.long).cuda()
        return self.criterion(logits,labels)

    def _dequeue_and_enqueue(self,keys,vals,cat, bs):
        if cat not in vals:
            return
        keys = keys[list(vals).index(cat)]
        batch_size = bs
        ptr = int(eval("self.ptr"+str(cat)))
        eval("self.queue"+str(cat))[:,ptr] = keys
        ptr = (ptr + batch_size) % self.queue_len
        eval("self.ptr"+str(cat))[0] = ptr

def get_seg_model(cfg, **kwargs):
    model = Seg_Resnet50_unet(cfg)
    return model