import gym
import torch
from pettingzoo.butterfly import knights_archers_zombies_v7
from env.adapter import create_kaz_env
from env.adapter import create_generals_env

class config():
    '''
        Dict that support "object.key" operator
    '''
    def __init__(self, cfg):
        self.add(cfg)

    def add(self, cfg:dict):
        for k, v in cfg.items():
            self.__setattr__(k, v)


def get_config(task = 'CartPole', exp_name = "default"):
    d = {}
    if task == 'CartPole':
        d = {
            # environment related
            'name':             'CartPole',
            'env':              lambda :gym.make('CartPole-v0'),
            'train_env_num':    50,
            'test_env_num':     10,

            # training related
            'buffer_size':      10000,
            'max_epoch':        30,
            'step_per_epoch':   1000,
            'step_per_collect': 500,
            'episode_per_test': 100,
            'batch_size':       256,
            'update_per_step':  0.1,
            'algo':             'ppo',
            'device':           'cuda:1' if torch.cuda.is_available() else 'cpu'
        }
    elif task == 'PettingZoo':
        d = {
            # environment related
            'name': 'PettingZoo',
            'env': lambda: create_kaz_env(),
            'train_env_num': 5,
            'test_env_num': 1,
            'num_agents':   2,

            # training related
            'buffer_size': 10000,
            'max_epoch': 30,
            'step_per_epoch': 1000,
            'step_per_collect': 500,
            'episode_per_test': 5,
            'batch_size': 16,
            'update_per_step': 0.2,
            'algo': 'ppo',
            'device': 'cuda:1' if torch.cuda.is_available() else 'cpu'
        }
    elif task == 'Generals':
        d = {
            # environment related
            'name': 'Generals',
            'env': lambda: create_generals_env(name = exp_name, auto_replay_id=False),
            'train_env': lambda: create_generals_env(name = exp_name, auto_replay_id=True),
            'valid_env': lambda: create_generals_env(name = exp_name),
            'train_env_num': 1,
            'test_env_num': 1,
            'num_agents':   2,

            # training related
            'buffer_size': 8192,
            'max_epoch': 100,
            'step_per_epoch': 640,
            'step_per_collect': 64,
            'episode_per_test': 1,
            'batch_size': 256,
            'update_per_step': 0.3,
            'algo': 'dqn',
            'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
        }

    return config(d)

if __name__ == '__main__':
    env = knights_archers_zombies_v7.env()
    env.reset()