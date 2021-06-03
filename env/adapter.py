import gym
import tianshou as ts
import torch, numpy as np
from torch import nn
from tianshou.data import Batch
from pettingzoo.butterfly import knights_archers_zombies_v7
from pettingzoo.utils import wrappers

from env.states import GeneralsMultiAgentEnv
from env.states import PlayerAction
from env.const import C


class KAZAdapter(knights_archers_zombies_v7.raw_env):#,gym.core.Env):
    '''
        Adapt knights env to tianshou training pipeline
    '''
    def __init__(self, **kwargs):
        super(KAZAdapter, self).__init__(**kwargs)
        # useful variable
        self.action_space = self.action_spaces[self.agents[0]]
        self.observation_space = self.observation_spaces[self.agents[0]]

    def reset(self):
        super(KAZAdapter, self).reset()
        self.name_to_id = {name: item+1 for item, name in enumerate(self.agents)}
        obs, _,_,_ = self.last()
        return Batch(obs = obs,agent_id = self.name_to_id[self.agent_selection])

    def step(self,action):
        super(KAZAdapter, self).step(action)
        obs, reward, done, info = self.last()
        obs = Batch(obs = obs,agent_id = self.name_to_id[self.agent_selection])
        return obs,np.repeat(np.array([reward]),self.num_agents,axis=0),done,info
    '''
    def reset(self):
        super(KAZAdapter, self).reset()
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
                super(KAZAdapter, self).step(action)
        return full_obs, full_rew, done, full_info
    '''

def create_kaz_env(**kwargs):
    env = KAZAdapter(num_knights =  0, **kwargs)
    return env

def ActionSpaceToActions(batch, H, W):
    acts = []
    for act in batch.act:
        h = act % (H*W) // W; w = act % W; half = act // (H*W) % 2; dir = act // (H*W) // 2
        acts.append(PlayerAction((h,w),C.MOVEABLE_DIRECTIONS[dir],half=half))
    return acts

class GeneralsAdapter(GeneralsMultiAgentEnv):
    def __init__(self, **kwargs):
        super(GeneralsAdapter, self).__init__(**kwargs)
        self.action_space = gym.spaces.discrete.Discrete(8*self.Hmax*self.Wmax)
        self.observation_space = gym.spaces.box.Box(low=-1,high=1,shape=(C.FEATURES*C.NUM_FRAME,self.Hmax,self.Wmax),dtype=np.float32)

    def reset(self):
        super(GeneralsAdapter, self).reset()
        self.name_to_id = {name: item+1 for item, name in enumerate(self.agents)}
        obs, _,_,_ = self.last()
        return Batch(obs = obs, agent_id = list(range(1,self.num_players+1)))

    def step(self, actions):
        actions = ActionSpaceToActions(actions, self.Hmax, self.Wmax)
        super(GeneralsAdapter, self).step(actions)
        obs, reward, done, info = self.last()
        # obs = Batch(obs = obs,agent_id = self.name_to_id[self.agent_selection])
        return obs,reward,done,info

if __name__=="__main__":
    act = [3, 9, 117]
    H = 10
    W = 10
    print(ActionSpaceToActions(Batch(act = act), H, W))