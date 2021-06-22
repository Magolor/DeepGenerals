import torch.distributions as dist
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from models.WindowAttention import SelfAttnBlock

model = SelfAttnBlock(128, (10,10), 32).to('cuda:1')
print(model)
x = torch.randn((1, 128, 80, 80)).to('cuda:1')
out = model(x)
print(out.size())


from timm.models.swin_transformer import swin_base_patch4_window12_384_in22k
