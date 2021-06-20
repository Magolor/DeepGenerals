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
from policy.mapolicy import MultiAgentPolicyManager



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


    if name == 'Generals':
        backbone = backbones.FCNBackbone(input_shape[0])

        actor_head = actionHead.actorHead((backbone.out_channels(), input_shape[1], input_shape[2]), action_space)
        actor = model.AdaptNetwork(actor_head, backbone).to(cfg.device)
        actor_optim = Adam(actor.parameters(),lr = 1e-6, weight_decay=5e-5)

        critic1_head = actionHead.actorHead((backbone.out_channels(), input_shape[1], input_shape[2]), action_space)
        critic2_head = actionHead.actorHead((backbone.out_channels(), input_shape[1], input_shape[2]), action_space)
        critic1 = model.CriticNetwork(critic1_head, backbone)
        critic2 = model.CriticNetwork(critic2_head, backbone)
        critic_optim1 = Adam(critic1.parameters(),lr = 1e-6, weight_decay=5e-5)
        critic_optim2 = Adam(critic2.parameters(),lr = 1e-6, weight_decay=5e-5)

        def single_policy(actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim):
            policy = ts.policy.SACPolicy(actor,actor_optim,critic1,critic1_optim,critic2,critic2_optim)
            return policy
            # multiagent

        policy = MultiAgentPolicyManager(
            [single_policy(actor,actor_optim,critic1,critic_optim1,critic2,critic_optim2) for _ in range(cfg.num_agents)]
        )


    return policy

