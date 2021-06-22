from torch import nn
import torch
from tianshou.data import Batch
from pettingzoo.butterfly import knights_archers_zombies_v7
import env.adapter as adapter
from torch.optim import Adam
from policy.ClipAdam import ClipAdam
from models import actionHead, backbones, model
import tianshou as ts
from itertools import chain
from torch.distributions import Categorical
from policy.mapolicy import MultiAgentPolicyManager



def get_ppo_policy(cfg, input_shape, action_space, name = 'CartPole'):
    policy = None
    if name == 'CartPole':
        actor = model.AdaptNetwork(actionHead.mlpHead(input_shape,action_space,hidden_dim=128,layer=3)).to(cfg.device)
        critic = model.CriticNetwork(actionHead.mlpHead(input_shape, 1, hidden_dim=128, layer=3)).to(cfg.device)
        optim = ClipAdam(chain(actor.parameters(), critic.parameters()),clip_value=5,lr = 1e-4)
        ts.policy.PPOPolicy(actor,critic,optim = optim)
        policy = ts.policy.PPOPolicy(actor,critic,optim,dist_fn=lambda logits:Categorical(logits=logits))

    if name == 'Generals':
        backbone = backbones.FCNBackbone(input_shape[0])
        critic_head = actionHead.criticHead((backbone.out_channels(), input_shape[1],input_shape[2]), action_space)
        actor_head = actionHead.actorHead((backbone.out_channels(), input_shape[1],input_shape[2]), action_space)
        actor = model.AdaptNetwork(actor_head, backbone).to(cfg.device)
        critic = model.CriticNetwork(critic_head,backbone).to(cfg.device)
        optim = ClipAdam(chain(backbone.parameters(),actor_head.parameters(),critic_head.parameters()),
                                 lr=1e-6, weight_decay=5e-5,clip_value=5)

        def single_policy(actor, critic, optim):
            policy = ts.policy.PPOPolicy(actor,critic, optim, dist_fn=lambda logits:Categorical(logits=torch.clip(logits,max=50)))
            return policy
            # multiagent

        policy = MultiAgentPolicyManager(
            [single_policy(actor, critic, optim) for _ in range(cfg.num_agents)]
        )

    return policy

