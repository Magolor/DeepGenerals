import gym
import tianshou as ts
import torch, numpy as np
from torch import nn
from tianshou.data import Batch
from pettingzoo.butterfly import knights_archers_zombies_v7
from pettingzoo.utils import wrappers



class MultiAgentAdapter(knights_archers_zombies_v7.raw_env):#,gym.core.Env):
    '''
        Adapt knights env to tianshou training pipeline
    '''
    def __init__(self, **kwargs):
        super(MultiAgentAdapter, self).__init__(**kwargs)
        # useful variable
        self.action_space = self.action_spaces[self.agents[0]]
        self.observation_space = self.observation_spaces[self.agents[0]]

    def reset(self):
        super(MultiAgentAdapter, self).reset()
        self.name_to_id = {name: item+1 for item, name in enumerate(self.agents)}
        obs, _,_,_ = self.last()
        return Batch(obs = obs,agent_id = self.name_to_id[self.agent_selection])

    def step(self,action):
        super(MultiAgentAdapter, self).step(action)
        obs, reward, done, info = self.last()
        obs = Batch(obs = obs,agent_id = self.name_to_id[self.agent_selection])
        return obs,np.repeat(np.array([reward]),self.num_agents,axis=0),done,info
    '''
    def reset(self):
        super(MultiAgentAdapter, self).reset()
        self.name_to_id = {name: item for item, name in enumerate(self.agents)}
        # first iteration random
        actions = self.action_space.sample()
        obs, rew, done, info = self.step(actions)
        return obs
   
    def step(self, actions):
        full_obs = {}
        full_rew = {}
        full_info = {}
        done = False
        for id,agent in enumerate(self.agent_iter(self.num_agents)):
            agent_id = f"agent_{self.name_to_id[agent]}"
            full_obs[agent_id], full_rew[agent_id], done, full_info[agent_id]= self.last()
            if not done:
                action = actions[id]
                super(MultiAgentAdapter, self).step(action)
        return full_obs, full_rew, done, full_info
    '''

def create_knight_env(**kwargs):
    env = MultiAgentAdapter(num_knights =  0, **kwargs)
    return env
