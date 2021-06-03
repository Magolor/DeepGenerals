import torch, numpy as np
from torch import nn
from timm.models.densenet import densenet121
from timm.models.resnet import resnet50
from torchvision.models.segmentation.segmentation import IntermediateLayerGetter
import torch.nn.functional as F

# base class
class actionHead(nn.Module):
    '''
        adaptive action predictor
    '''
    def __init__(self, input_shape, action_space):
        super(actionHead, self).__init__()
        self.input_shape = input_shape
        self.action_space = action_space

    def preprocess(self, x, device):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x,device= device, dtype=torch.float)
        return x

    def forward(self,x, **kwargs):
        pass

class spacePreservedHead(actionHead):
    '''
       preserve space info -> basic flatten
       (B, C, H, W) -> (B, action_space, H, W) -> (B, action_space * H * W)
    '''
    def __init__(self,  input_shape, action_space):
        super(actionHead, self).__init__(input_shape, action_space//(input_shape[1]*input_shape[2]))
        self.cls = nn.Conv2d(input_shape[0], self.action_space,1)

    def forward(self, x, **kwargs):
        x = self.preprocess(x, self.cls.weight.device)
        logits = torch.flatten(self.cls(x),start_dim=1)
        return logits

class mlpHead(actionHead):
    '''
       feature_vector (B, C) -> (B, action_space)
    '''
    def __init__(self, input_shape, action_space, hidden_dim = 256, layer = 2):
        super(mlpHead, self).__init__(input_shape, action_space)
        self.layer = layer
        self.mlp = nn.Sequential()
        for i in range(layer):
            in_dim = input_shape[0] if i == 0 else hidden_dim
            self.mlp.add_module(f'linear{i+1}',nn.Linear(in_dim,hidden_dim))
            self.mlp.add_module(f'relu{i+1}',nn.ReLU(True))
        in_dim = input_shape[0] if layer==0 else hidden_dim
        self.cls =  nn.Linear(in_dim, self.action_space)

    def forward(self, x, **kwargs):
        x = self.preprocess(x, self.cls.weight.device)
        if self.layer>0:
            x = self.mlp(x)
        logits = self.cls(x)
        return logits


if __name__ == '__main__':
    model = mlpHead(20)
    print(model)
    x = torch.randn((1,100))
    out = model(x)
    print(out.size())