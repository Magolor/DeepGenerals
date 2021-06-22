from torch.optim import Adam
import torch
from torch.nn.utils import clip_grad_value_

class ClipAdam(Adam):
    def __init__(self,params, clip_value=5, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        super(ClipAdam, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)
        self.params = params
        self.clip_value = clip_value

    @torch.no_grad()
    def step(self, closure=None):
        clip_grad_value_(self.params,clip_value=self.clip_value)
        return super(ClipAdam, self).step(closure)
