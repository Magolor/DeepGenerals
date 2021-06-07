import gym
import tianshou as ts
import torch, numpy as np
from torch import nn
from tianshou.data import Batch
from pettingzoo.butterfly import knights_archers_zombies_v7
import env.adapter as adapter
from models.backbones import backbone

class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(*[
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape))
        ])

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.model[0].weight.device)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1)).cpu()
        return logits, state

def train():
    env = gym.make('CartPole-v0')
    # create environment
    train_envs = ts.env.DummyVectorEnv([lambda: gym.make('CartPole-v0') for _ in range(10)])
    test_envs = ts.env.DummyVectorEnv([lambda: gym.make('CartPole-v0') for _ in range(100)])

    # crate a network
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    net = Net(state_shape, action_shape).to('cuda:1')
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)

    # create a policy
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

def eval_dqn():
    env = adapter.Create_env()
    policy = torch.load('dqn.pt')
    policy.eval()

    done = False
    info = {}
    obs = env.reset()
    while not done:
        obs.obs = np.expand_dims(obs.obs,axis = 0)
        action = policy(Batch(obs=obs.obs, info=info)).act[0]
        env.render()
        obs, _, done, _ = env.step(action)

if __name__ == "__main__":
    train()