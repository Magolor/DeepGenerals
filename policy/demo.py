import gym
import tianshou as ts
import torch, numpy as np
from torch import nn
from tianshou.data import Batch
from pettingzoo.butterfly import knights_archers_zombies_v7
import env.adapter as adapter
from models.backbones import backbone

# for discrete environment
class demo_net(nn.Module):
    def __init__(self, action_shape, state_shape):
        super().__init__()
        self.model = nn.Sequential(*[
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape))
        ])

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1)).cpu()
        return logits, state

def demo(env_name, net_cls:nn.Module, policy_cls:ts.policy.BasePolicy):
    '''
    Args:
        env_name: gym name
        net_cls: backbone network class
        policy_cls:  policy class
    '''
    env = gym.make(env_name)
    # create environment
    train_envs = ts.env.DummyVectorEnv([lambda: gym.make(env_name) for _ in range(10)])
    test_envs = ts.env.DummyVectorEnv([lambda: gym.make(env_name) for _ in range(100)])

    # crate a network
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    net = net_cls(action_shape, state_shape)
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)

    # create a policy
    ts.policy.DQNPolicy()
    policy = policy_cls(net, optim, )
    policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.99, estimation_step=3, target_update_freq=320)
    # create a collector
    train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(20000, 10), exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

    # create a trainer
    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=10, step_per_epoch=10000, step_per_collect=16,
        update_per_step=0.1, episode_per_test=2000, batch_size=64,
        train_fn=lambda epoch, env_step: policy.set_eps(0.1),
        test_fn=lambda epoch, env_step: policy.set_eps(0.05),
        stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold)
    print(f'Finished training! Use {result["duration"]}')

    # save
    torch.save(policy, 'demo.pt')

def eval():
    env = gym.make('CartPole-v0')
    policy = torch.load('demo.pt')
    policy.eval()
    policy.set_eps(0.05)

    done = False
    info = {}
    obs = env.reset()
    while not done:
        action = policy(Batch(obs = [obs], info = info)).act[0]
        env.render()
        obs, _, done, _ = env.step(action)
