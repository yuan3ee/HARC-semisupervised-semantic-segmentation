from torch.autograd import Variable
import torch.nn as nn


class s4GAN_discriminator(nn.Module):

    def __init__(self, num_classes, ndf = 64):
        super(s4GAN_discriminator, self).__init__()

        self.conv1 = nn.Conv2d(num_classes+3, ndf, kernel_size=4, stride=2, padding=1) # 256 x 256
        self.conv2 = nn.Conv2d(  ndf, ndf*2, kernel_size=4, stride=2, padding=1) # 128 x 128
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1) # 64 x 64
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1) # 32 x 32
        # if dataset == 'pascal_voc' or dataset == 'pascal_context':
        #     self.avgpool = nn.AvgPool2d((20, 20))
        # elif dataset == 'cityscapes':
        #     self.avgpool = nn.AvgPool2d((16, 32))
        self.avgpool = nn.AvgPool2d((32, 32))
        self.fc = nn.Linear(ndf*8, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.drop = nn.Dropout2d(0.5)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
       
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
       
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
        
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
        
        x = self.conv4(x)
        x = self.leaky_relu(x)
        
        maps = self.avgpool(x)
        conv4_maps = maps 
        out = maps.view(maps.size(0), -1)
        out = self.sigmoid(self.fc(out))
        
        return out, conv4_maps


def get_seg_model(cfg, **kwargs):
    model = s4GAN_discriminator(num_classes=cfg.DATASET.NUM_CLASSES)
    if cfg.MODEL.PRETRAINED_D:
        saved_state_dict = torch.load(cfg.MODEL.PRETRAINED_D)
        new_params = model.state_dict().copy()
        for name, param in new_params.items():
            if name in saved_state_dict and param.size() == saved_state_dict[name].size():
                print(1)
                new_params[name].copy_(saved_state_dict[name])
        model.load_state_dict(new_params)

    return model