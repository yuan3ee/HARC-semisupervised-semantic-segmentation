from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# import our stuffs
from .bn_helper import BatchNorm2d, BatchNorm2d_class, relu_inplace


class discriminator_DUL(nn.Module):

    def __init__(self, cfg, ndf = 64):
        super(discriminator_DUL, self).__init__()
        self.cfg = cfg
        self.drop = nn.Dropout2d(0.5)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=relu_inplace)
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AvgPool2d((64, 64))
        self.conv = nn.Conv2d(self.cfg.DATASET.NUM_CLASSES+3, ndf, kernel_size=3, stride=1, padding=1)  # 512 x 512
        self.bn = BatchNorm2d(ndf)
        self.conv1 = nn.Conv2d(ndf, ndf*2, kernel_size=3, stride=2, padding=1)  # 256 x 256
        self.bn1 = BatchNorm2d(ndf*2)
        self.conv2 = nn.Conv2d(ndf*2, ndf*4, kernel_size=3, stride=2, padding=1)  # 128 x 128
        self.bn2 = BatchNorm2d(ndf*4)
        self.conv3 = nn.Conv2d(ndf*4, ndf*8, kernel_size=3, stride=2, padding=1)  # 64 x 64
        self.bn3 = BatchNorm2d(ndf*8)
        self.head = nn.Conv2d(ndf*8, 1, kernel_size=3, stride=1, padding=1)  # 64 x 64
        # self._init_weights()


    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, BatchNorm2d_class):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        _, _, h, w = x.size()
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x_down2x = self.leaky_relu(x)
        x = self.conv2(x_down2x)
        x = self.bn2(x)
        x_down4x = self.leaky_relu(x)
        x = self.conv3(x_down4x)
        x = self.bn3(x)
        x_down8x = self.leaky_relu(x)
        x_down8x = self.head(x_down8x)

        out_CR = self.avgpool(x_down8x)

        out = F.interpolate(x_down8x, (h, w), mode='bilinear', align_corners=True)
        out = self.sigmoid(out)

        return out_CR, out


def get_seg_model(cfg, **kwargs):
    model = discriminator_DUL(cfg)
    if cfg.MODEL.PRETRAINED_D:
        saved_state_dict = torch.load(cfg.MODEL.PRETRAINED_D)
        new_params = model.state_dict().copy()
        for name, param in new_params.items():
            if name in saved_state_dict and param.size() == saved_state_dict[name].size():
                print(1)
                new_params[name].copy_(saved_state_dict[name])
        model.load_state_dict(new_params)

    return model