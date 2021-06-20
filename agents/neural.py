from .base import *
import torch
from tianshou.data import Batch

class NeuralAgent(BaseAgent):
    def __init__(self, **kwargs):
        super(NeuralAgent, self).__init__()
        path = kwargs.pop('path')
        device = kwargs.pop('device')
        self.dqn = torch.load(path, device).policies[0]

    def get_action(self, obs, **info):
        # return int type or None
        net_batch = Batch({
            'obs': obs.obs,
            'done': False,
            'info': {},
            'policy': Batch(),
            'rew': Batch(),
            'act': Batch(),
            'obs_next': Batch(),
            'board': obs.board
        })
        out = self.dqn.forward(net_batch)
        return out.act[0]
