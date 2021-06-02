import gym
import tianshou as ts
from tianshou.utils.log_tools import BasicLogger, SummaryWriter
import torch, numpy as np
from policy.policy import get_policy, ExplorationRateDecayPolicy
from pettingzoo.butterfly import knights_archers_zombies_v7
from env.adapter import create_knight_env
from tianshou.data import Batch
from tool import utils
import config
import json


def train(cfg, log_dir = None):
    # get environment
    print("Create Virtual Environment:")
    env = cfg.env()
    train_envs = ts.env.SubprocVectorEnv([cfg.env for _ in range(cfg.train_env_num)])
    test_envs = ts.env.SubprocVectorEnv([cfg.env for _ in range(cfg.test_env_num)])
    print('Done!')
    # crate an agent (policy)
    print("Create Agent:")
    input_shape = env.observation_space.shape or env.observation_space.n
    n = env.action_space.shape or env.action_space.n
    policy = get_policy(cfg, input_shape, n, cfg.name)
    print('Done!')
    # create a collector
    print("Create Buffer:")
    train_collector = ts.data.Collector(policy, train_envs,
                                        ts.data.VectorReplayBuffer(cfg.buffer_size, cfg.train_env_num),
                                        exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)
    print("Done!")
    # create a trainer
    print("Start training!")
    result = ts.trainer.offpolicy_trainer(
        policy = policy,
        train_collector = train_collector,
        test_collector = test_collector,
        max_epoch=cfg.max_epoch,
        step_per_epoch=cfg.step_per_epoch,
        step_per_collect=cfg.step_per_collect,
        episode_per_test=cfg.episode_per_test,
        batch_size=cfg.batch_size,
        update_per_step=cfg.update_per_step,
        train_fn=ExplorationRateDecayPolicy(policy, cfg.max_epoch,cfg.step_per_epoch),
        test_fn=ExplorationRateDecayPolicy(policy, cfg.max_epoch,cfg.step_per_epoch,mile_stones=(),rates=(0.05,)),
        save_fn=lambda policy: torch.save(policy,utils.get_fs().get_checkpoint_dirpath()/'best.pt'),
        logger=BasicLogger(SummaryWriter(log_dir))
    )
    utils.log(result)
    #with open(utils.get_fs().get_root_path()/'data.json', 'w') as f:
    #    json.dump(f,result.items())
    # save
    torch.save(policy, cfg.name+'.pt')


def visualize(cfg, model_path=None, random = False):
    env = cfg.env()
    if not random:
        if model_path is None:
            model_path = cfg.name + '.pt'
        policy = torch.load(model_path)
        policy.eval()

    for i in range(100):
        done = False
        info = {}
        obs = env.reset()
        while not done:
            # for multi agent
            if isinstance(obs, Batch):
                obs = obs.obs
            obs = np.expand_dims(obs,axis = 0)
            if random:
                action = env.action_space.sample()
            else:
                action = policy(Batch(obs=obs, info=info)).act[0]
            env.render()
            obs, reward, done, info = env.step(action)

if __name__ == '__main__':
    '''
    'CartPole', 'PettingZoom'
    '''
    cfg = config.get_config('PettingZoom')
    utils.init('Test', 'Experiment')
    train(cfg, utils.get_fs().get_root_path())

    #visualize(cfg, random=True)