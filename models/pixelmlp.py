import torch.nn as nn

class PixelMlp(nn.Module):
    """ Multilayer perceptron per pixel."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features,hidden_features,1)
        self.act = act_layer()
        self.bn1 = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features,1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

