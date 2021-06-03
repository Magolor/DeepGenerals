import torch
import numpy as np
from torch import nn
from timm.models.densenet import densenet121
from timm.models.resnet import resnet34
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torchvision.models.segmentation.segmentation import IntermediateLayerGetter
import torch.nn.functional as F
from torchvision.models.densenet import _DenseBlock,_Transition
from models.WindowAttention import SelfAttnBlock
from models.pixelmlp import PixelMlp


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

class CBR(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(CBR, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=1, stride=1, bias=False))


class FCNBackbone(backbone):
    def __init__(self, in_channels):
        super(FCNBackbone, self).__init__(False, False)
        self.stem = PixelMlp(in_channels,128, 128)
        self.stage1 = _DenseBlock(4, 128, 4 ,32 ,0 ,True)
        channels = 128 + 4*32
        self.cbr = CBR(channels, channels)
        self.stage2 = _DenseBlock(8,channels,4,32,0,True)
        self.channels = channels + 8 * 32

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(self.cbr(self.stage1(x)))
        return x

    def get_device(self):
        return self.cbr.conv.weight.device

    def out_channels(self):
        return self.channels

    def out_stride(self):
        return 1

class SelfAttnBackbone(backbone):
    '''
        Fully preserve spacial information
        (B, C, H, W) -> (B, C', H, W).
    '''
    def __init__(self, in_channels):
        super(SelfAttnBackbone, self).__init__()
        self.stem = PixelMlp(in_channels, )



if __name__ == '__main__':
    model = SelfAttnBlock(128,(10,10),32)
    print(model)
    x = torch.randn((1,3,80,80))
    out = model(x)
    print(out.size())