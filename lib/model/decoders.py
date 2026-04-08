import math , time
import torch
import torch.nn.functional as F
from torch import nn
import contextlib
import random
import numpy as np
import cv2
from torch.distributions.uniform import Uniform


upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)


class DropOutDecoder(nn.Module):
    def __init__(self, drop_rate=0.3, spatial_dropout=True):
        super(DropOutDecoder, self).__init__()
        self.dropout = nn.Dropout2d(p=drop_rate) if spatial_dropout else nn.Dropout(drop_rate)

    def forward(self, x, _):
        x = self.dropout(x)
        x = upsample(x)
        return x


class FeatureDropDecoder(nn.Module):
    def __init__(self):
        super(FeatureDropDecoder, self).__init__()

    def feature_dropout(self, x):
        attention = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
        threshold = max_val * np.random.uniform(0.7, 0.9)
        threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        return x.mul(drop_mask)

    def forward(self, x, _):
        x = self.feature_dropout(x)
        x = upsample(x)
        return x


class FeatureNoiseDecoder(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoiseDecoder, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x, _):
        x = self.feature_based_noise(x)
        x = upsample(x)
        return x



def _l2_normalize(d):
    # Normalizing per batch axis
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def get_r_adv(x, it=1, xi=1e-1, eps=10.0):
    """
    Virtual Adversarial Training
    https://arxiv.org/abs/1704.03976
    """
    x_detached = x.detach()
    with torch.no_grad():
        pred = F.softmax(upsample(x_detached), dim=1)

    d = torch.rand(x.shape).sub(0.5).to(x.device)
    d = _l2_normalize(d)

    for _ in range(it):
        d.requires_grad_()
        pred_hat = upsample(x_detached + xi * d)
        logp_hat = F.log_softmax(pred_hat, dim=1)
        adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
        adv_distance.backward()
        d = _l2_normalize(d.grad)
        upsample.zero_grad()

    r_adv = d * eps
    return r_adv


class VATDecoder(nn.Module):
    def __init__(self, xi=1e-1, eps=10.0, iterations=1):
        super(VATDecoder, self).__init__()
        self.xi = xi
        self.eps = eps
        self.it = iterations

    def forward(self, x, _):
        r_adv = get_r_adv(x, self.it, self.xi, self.eps)
        x = upsample(x + r_adv)
        return x


def guided_cutout(output, resize, erase=0.4, use_dropout=False):
    if len(output.shape) == 3:
        masks = (output > 0).float()
    else:
        masks = (output.argmax(1) > 0).float()

    if use_dropout:
        p_drop = random.randint(3, 6)/10
        maskdroped = (F.dropout(masks, p_drop) > 0).float()
        maskdroped = maskdroped + (1 - masks)
        maskdroped.unsqueeze_(0)
        maskdroped = F.interpolate(maskdroped, size=resize, mode='nearest')

    masks_np = []
    for mask in masks:
        mask_np = np.uint8(mask.cpu().numpy())
        mask_ones = np.ones_like(mask_np)
        try: # Version 3.x
            _, contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except: # Version 4.x
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polys = [c.reshape(c.shape[0], c.shape[-1]) for c in contours if c.shape[0] > 50]
        for poly in polys:
            min_w, max_w = poly[:, 0].min(), poly[:, 0].max()
            min_h, max_h = poly[:, 1].min(), poly[:, 1].max()
            bb_w, bb_h = max_w-min_w, max_h-min_h
            rnd_start_w = random.randint(0, int(bb_w*(1-erase)))
            rnd_start_h = random.randint(0, int(bb_h*(1-erase)))
            h_start, h_end = min_h+rnd_start_h, min_h+rnd_start_h+int(bb_h*erase)
            w_start, w_end = min_w+rnd_start_w, min_w+rnd_start_w+int(bb_w*erase)
            mask_ones[h_start:h_end, w_start:w_end] = 0
        masks_np.append(mask_ones)
    masks_np = np.stack(masks_np)

    maskcut = torch.from_numpy(masks_np).float().unsqueeze_(1)
    maskcut = F.interpolate(maskcut, size=resize, mode='nearest')

    if use_dropout:
        return maskcut.to(output.device), maskdroped.to(output.device)
    return maskcut.to(output.device)


class CutOutDecoder(nn.Module):
    def __init__(self, erase=0.4):
        super(CutOutDecoder, self).__init__()
        self.erase = erase

    def forward(self, x, pred=None):
        maskcut = guided_cutout(pred, erase=self.erase, resize=(x.size(2), x.size(3)))
        x = x * maskcut
        x = upsample(x)
        return x


def guided_masking(x, output, resize, return_msk_context=True):
    if len(output.shape) == 3:
        masks_context = (output > 0).float().unsqueeze(1)
    else:
        masks_context = (output.argmax(1) > 0).float().unsqueeze(1)
    
    masks_context = F.interpolate(masks_context, size=resize, mode='nearest')

    x_masked_context = masks_context * x
    if return_msk_context:
        return x_masked_context

    masks_objects = (1 - masks_context)
    x_masked_objects = masks_objects * x
    return x_masked_objects


class ContextMaskingDecoder(nn.Module):
    def __init__(self):
        super(ContextMaskingDecoder, self).__init__()

    def forward(self, x, pred=None):
        x_masked_context = guided_masking(x, pred, resize=(x.size(2), x.size(3)), return_msk_context=True)
        x_masked_context = upsample(x_masked_context)
        return x_masked_context


class ObjectMaskingDecoder(nn.Module):
    def __init__(self):
        super(ObjectMaskingDecoder, self).__init__()

    def forward(self, x, pred=None):
        x_masked_obj = guided_masking(x, pred, resize=(x.size(2), x.size(3)), return_msk_context=False)
        x_masked_obj = upsample(x_masked_obj)

        return x_masked_obj

