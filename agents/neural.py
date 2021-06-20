from .base import *
import torch
import numpy as np
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
        raw_out = self.dqn.forward(net_batch)
        act_list = obs.board.GetPlayerState(self.agent_id).AvailableActions(serialize=True)
        mask = [(i in act_list) for i in range(np.prod(obs.board.board_shape)*8)]
        logit = raw_out.logits.cpu() * torch.tensor(mask,dtype=torch.float)
        act = torch.argmax(logit,dim = 1).cpu().numpy()
        return act[0]
