import gym
import torch
from pettingzoo.butterfly import knights_archers_zombies_v7
from env.adapter import create_knight_env

class config():
    '''
        Dict that support "object.key" operator
    '''
    def __init__(self, cfg):
        self.add(cfg)

    def add(self, cfg:dict):
        for k, v in cfg.items():
            self.__setattr__(k, v)


def get_config(task = 'CartPole'):
    d = {}
    if task == 'CartPole':
        d = {
            # environment related
            'name':             'CartPole',
            'env':              lambda :gym.make('CartPole-v0'),
            'train_env_num':    10,
            'test_env_num':     2,

            # training related
            'buffer_size':      10000,
            'max_epoch':        10,
            'step_per_epoch':   10000,
            'step_per_collect': 50,
            'episode_per_test': 100,
            'batch_size':       64,
            'update_per_step':  0.3,
            'algo':             'dqn',
            'device':           'cuda:1' if torch.cuda.is_available() else 'cpu'
        }
    elif task == 'PettingZoom':
        d = {
            # environment related
            'name': 'PettingZoom',
            'env': lambda: create_knight_env(),
            'train_env_num': 1,
            'test_env_num': 1,
            'num_agents':   2,

            # training related
            'buffer_size': 1000,
            'max_epoch': 10,
            'step_per_epoch': 50,
            'step_per_collect': 50,
            'episode_per_test': 1,
            'batch_size': 64,
            'update_per_step': 0.3,
            'algo': 'dqn',
            'device': 'cuda:1' if torch.cuda.is_available() else 'cpu'
        }

    return config(d)

if __name__ == '__main__':
    env = knights_archers_zombies_v7.env()
    env.reset()