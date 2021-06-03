from torch import nn
from tianshou.data import Batch
from pettingzoo.butterfly import knights_archers_zombies_v7
import env.adapter as adapter
from torch.optim import Adam
from models import actionHead, backbones, model
import tianshou as ts
from itertools import chain
from torch.distributions import Categorical



def get_sac_policy(cfg, input_shape, action_space, name='CartPole'):
    policy = None
    if name == 'CartPole':
        actor = model.AdaptNetwork(actionHead.mlpHead(input_shape, action_space, hidden_dim=128, layer=3)).to(
            cfg.device)
        critic1 = model.AdaptNetwork(actionHead.mlpHead(input_shape, 1, hidden_dim=128, layer=3)).to(cfg.device)
        critic2 = model.AdaptNetwork(actionHead.mlpHead(input_shape, 1, hidden_dim=128, layer=3)).to(cfg.device)

        actor_optim = Adam(actor.parameters(),lr = 1e-3)
        critic1_optim = Adam(critic1.parameters(), lr = 1e-3)
        critic2_optim = Adam(critic2.parameters(), lr=1e-3)
        policy = ts.policy.SACPolicy(actor,actor_optim,critic1,critic1_optim,critic2,critic2_optim)
    return policy

