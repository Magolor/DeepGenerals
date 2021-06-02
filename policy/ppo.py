from torch import nn
from tianshou.data import Batch
from pettingzoo.butterfly import knights_archers_zombies_v7
import env.adapter as adapter
from torch.optim import Adam
from models import actionHead, backbones, model
import tianshou as ts
from itertools import chain



def get_ppo_policy(cfg, input_shape, action_space, name = 'CartPole'):
    policy = None
    if name == 'CartPole':
        actor = model.AdaptNetwork(actionHead.mlpHead(input_shape,action_space,hidden_dim=128,layer=3)).to(cfg.device)
        critic = model.AdaptNetwork(actionHead.mlpHead(input_shape, 1, hidden_dim=128, layer=3)).to(cfg.device)
        optim = chain(actor.parameters(), critic.parameters())
        #ts.policy.PPOPolicy(actor,critic,optim,dist_fn)

    return policy

