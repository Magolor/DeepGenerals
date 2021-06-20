import gym
import tianshou as ts
import torch, numpy as np
from torch import nn
from tianshou.data import Batch
from pettingzoo.butterfly import knights_archers_zombies_v7
from pettingzoo.utils import wrappers
from gym.spaces import Discrete,MultiDiscrete
from env.generalsio import GeneralsMultiAgentEnv
from env.states import PlayerActionFromID
from env.const import C
from utils import *


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

def ActionSpaceToActions(batch, W, H):
    return [PlayerActionFromID(act, (W,H)) for act in batch]

class GeneralsAdapter(GeneralsMultiAgentEnv):
    def __init__(self, **kwargs):
        super(GeneralsAdapter, self).__init__(**kwargs)
        self.action_space = gym.spaces.discrete.Discrete(8*self.Wmax*self.Hmax)
        self.observation_space = gym.spaces.box.Box(low=-1,high=1,shape=(C.FEATURES*C.NUM_FRAME,self.Wmax,self.Hmax),dtype=np.float32)
        self.name = kwargs['name']; self.auto_replay_id = kwargs['auto_replay_id']

    def reset(self, replay_id=None):
        if replay_id is None and self.auto_replay_id:
            replay_id = self.name+"_"+DATETIME()+"_"+RANDSTRING(8)
        super(GeneralsAdapter, self).reset(replay_id); obs, _, _, info = self.last(); god = info.pop('god'); board = info.pop('board')
        return Batch(obs = [torch.cat([f.serialize() for f in o],dim=0) for o in obs], agent_id = list(range(1,self.num_players+1)), god = god, board = board,wtf = 0)

    def step(self, actions):
        actions = ActionSpaceToActions(actions, self.Wmax, self.Hmax)
        super(GeneralsAdapter, self).step(actions)
        obs, reward, done, info = self.last(); god = info.pop('god'); board = info.pop('board')
        return Batch(obs = [torch.cat([f.serialize() for f in o],dim=0) for o in obs], agent_id = list(range(1,self.num_players+1)), god = god, board = board, wtf = 0), reward, done, info

def create_generals_env(name = "default", auto_replay_id = True):
    env = GeneralsAdapter(name = name, auto_replay_id = auto_replay_id)
    return env

VALID_ENV_IDS = {"kaz": create_kaz_env, "generals": create_generals_env}
def env(ID, **kwargs):
    assert(ID in VALID_ENV_IDS); return VALID_ENV_IDS[ID](**kwargs)

if __name__=="__main__":
    pass
    # act = [3, 9, 117, 593]
    # H = 10
    # W = 10
    # for act in ActionSpaceToActions(Batch(act = act), H, W):
    #     print(act)