import torch
import numpy as np
from torch import nn
from timm.models.densenet import densenet121
from timm.models.resnet import resnet34
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torchvision.models.segmentation.segmentation import IntermediateLayerGetter
import torch.nn.functional as F


class backbone(nn.Module):
    '''
    a convolution module: (B, C, H, W) -> (B, C', H', W')
    '''
    def __init__(self, flatten = False, channel_last=True):
        super(backbone, self).__init__()
        self.flatten = flatten
        self.channel_last = channel_last

    def preprocess(self, x, device):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x,dtype=torch.float)
        x = x.to(device)
        if self.channel_last:
            x = x.permute(0,3,1,2)
        return x

    def get_device(self):
        for w in self.backbone.parameters():
            return w.device

    def out_channels(self):
        pass

    def out_stride(self):
        pass

    def forward(self, obs):
        pass

class DenseBackbone(backbone):
    '''stride: 8, 16, 32'''
    def __init__(self, flatten, stride = 8, channel_last=True):
        super(DenseBackbone, self).__init__(flatten, channel_last)
        dense = densenet121(False)
        self.stride = stride
        self.backbone = IntermediateLayerGetter(dense.features,return_layers={f'denseblock{int(np.log2(stride)-1)}':'out'})

    def out_channels(self):
        return self.stride * 64

    def out_stride(self):
        return self.stride

    def forward(self, x):
        x = self.preprocess(x, self.get_device())
        x = self.backbone(x)['out']
        if self.flatten:
            x = F.adaptive_avg_pool2d(x,(1,1))
            x = torch.flatten(x, start_dim=1)
        return x

class ResnetBackbone(backbone):
    def __init__(self, flatten, stride=8, channel_last=True):
        super(ResnetBackbone, self).__init__(flatten, channel_last)
        res = resnet34(False)
        self.stride = stride
        self.backbone = IntermediateLayerGetter(res, return_layers={f'layer{int(np.log2(stride)-1)}':'out'})

    def out_channels(self):
        return self.stride * 16

    def out_stride(self):
        return self.stride

    def forward(self, x):
        x = self.preprocess(x, self.get_device())
        x = self.backbone(x)['out']
        if self.flatten:
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, start_dim=1)
        return x

class SelfAttnBackbone(backbone):
    '''
        Fully preserve spacial information
        (B, C, H, W) -> (B, C', H, W).
    '''
    def __init__(self, stride):
        super(SelfAttnBackbone, self).__init__()
        pass



if __name__ == '__main__':
    model = ResnetBackbone(False,32)
    print(model)
    x = torch.randn((1,3,64,64))
    out = model(x)
    print(out.size())