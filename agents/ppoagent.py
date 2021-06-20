from .base import *
import torch
from pathlib import Path
from tianshou.data import Batch
from tianshou.policy import PPOPolicy
from torch.distributions import Categorical
import numpy as np

class PPOAgent(BaseAgent):
    def __init__(self, **kwargs):
        super(PPOAgent, self).__init__()
        path = Path(kwargs.pop('path'))
        actor_path = path/('actor_'+path.name+'.pt')
        critic_path = path/('critic_'+path.name+'.pt')
        device = kwargs.pop('device')
        actor = torch.load(actor_path, device)
        critic = torch.load(critic_path, device)
        optim = torch.optim.Adam(actor.parameters(), lr=1e-6, weight_decay=5e-5)
        self.ppo =  PPOPolicy(actor, critic, optim, dist_fn=lambda logits:Categorical(logits=logits))

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
        raw_out = self.ppo.forward(net_batch)
        act_list = obs.board.GetPlayerState(self.agent_id).AvailableActions(serialize=True)
        mask = [(i in act_list) for i in range(np.prod(obs.board.board_shape) * 8)]
        logit = raw_out.logits.cpu() * torch.tensor(mask, dtype=torch.float)
        act = torch.argmax(logit, dim=1).cpu().numpy()
        return act[0]
