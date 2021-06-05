from torch import nn
from tianshou.data import Batch
from pettingzoo.butterfly import knights_archers_zombies_v7
import env.adapter as adapter
from torch.optim import SGD, Adam
from models import actionHead, backbones, model
import tianshou as ts
from policy.mapolicy import MultiAgentPolicyManager


def get_dqn_policy(cfg, input_shape, action_space, name = 'CartPole'):
    policy = None
    if name == 'CartPole':
        net = model.AdaptNetwork(actionHead.mlpHead(input_shape,action_space,hidden_dim=64,layer=2)).to(cfg.device)
        optim = Adam(net.parameters(),lr=1e-4,weight_decay=1e-4)
        policy = ts.policy.DQNPolicy(net, optim, estimation_step=3,target_update_freq=320)

    if name == 'PettingZoo':
        def single_policy():
            backbone = backbones.ResnetBackbone(flatten=True, stride=32, channel_last=True)
            head = actionHead.mlpHead((backbone.out_channels(),-1,-1), action_space,layer=0)
            net = model.AdaptNetwork(head, backbone).to(cfg.device)
            optim = Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
            policy = ts.policy.DQNPolicy(net, optim, estimation_step=20, target_update_freq=200)
            return policy
        # multiagent
        policy = MultiAgentPolicyManager(
            [single_policy() for _ in range(cfg.num_agents)]
        )

    if name == 'Generals':
        def single_policy():
            backbone = backbones.FCNBackbone(input_shape[0])
            head = actionHead.spacePreservedHead((backbone.out_channels(), input_shape[1],input_shape[2]), action_space)
            net = model.AdaptNetwork(head, backbone).to(cfg.device)
            optim = Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
            policy = ts.policy.DQNPolicy(net, optim, estimation_step=10, target_update_freq=200)
            return policy
            # multiagent
        policy = MultiAgentPolicyManager(
                [single_policy() for _ in range(cfg.num_agents)]
        )
    return policy

