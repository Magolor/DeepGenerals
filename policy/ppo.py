from torch import nn
import torch
from tianshou.data import Batch
from pettingzoo.butterfly import knights_archers_zombies_v7
import env.adapter as adapter
from torch.optim import Adam
from models import actionHead, backbones, model
import tianshou as ts
from itertools import chain
from torch.distributions import Categorical



def get_ppo_policy(cfg, input_shape, action_space, name = 'CartPole'):
    policy = None
    if name == 'CartPole':
        actor = model.AdaptNetwork(actionHead.mlpHead(input_shape,action_space,hidden_dim=128,layer=3)).to(cfg.device)
        critic = model.CriticNetwork(actionHead.mlpHead(input_shape, 1, hidden_dim=128, layer=3)).to(cfg.device)
        optim = torch.optim.Adam(chain(actor.parameters(), critic.parameters()),lr = 1e-4)
        ts.policy.PPOPolicy(actor,critic,optim = optim)
        policy = ts.policy.PPOPolicy(actor,critic,optim,dist_fn=lambda logits:Categorical(logits=logits))


    return policy

