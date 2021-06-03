from models import actionHead, backbones
import torch.nn as nn


class AdaptNetwork(nn.Module):
    '''
        a class to adapt API in "tianshou.policy"
    '''
    def __init__(self, head, backbone=None):
        super(AdaptNetwork, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, state = None, info = None):
        if self.backbone is not None:
            x = self.backbone(x)
        x = self.head(x)
        return x, None

class CriticNetwork(nn.Module):
    '''
          a class to adapt API in "tianshou.policy"
      '''

    def __init__(self, head, backbone=None):
        super(CriticNetwork, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, state=None, info=None):
        if self.backbone is not None:
            x = self.backbone(x)
        x = self.head(x)
        return x