import sys
import gym
sys.path.append("../")
from utils import *
from states import *

class GeneralsMultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(
        self,
        map0 = None,
        Ws = [5],
        Hs = [3],
        num_players = 2,
        p_mountain = 0.2,
        p_city = 0.05,
        army_generator = UniformArmyGenerator(25,101)
    ):
        super(GeneralsEnv, self).__init__()
        self.map = map0; self.state = None
        self.Ws = Ws; self.Hs = Hs
        self.num_players = num_players
        self.p_mountain = p_mountain
        self.p_city = p_city
        self.army_generator = army_generator
        self.history = []
    
    def step(self, actions):
        old_observations = [self.state.GetPlayerState(i) for i in range(self.num_players)]
        done = self.state.GetNextState_(actions); self.history.append(self.state.copy())
        new_observations = [new_state.GetPlayerState(i) for i in range(self.num_players)]
        rewards = [new_observations[i].Score() - old_observations[i].Score() for i in range(self.num_players)]
        info = {}
        return new_observations, rewards, done, info

    def reset(self):
        self.state = NewBoardStateFromMap(
            NewRandomMap(
                np.random.choice(self.Ws),
                self.num_players,
                np.random.choice(self.Hs),
                self.p_mountain,
                self.p_city,
            )
            if self.map is None else self.map,
            army_generator = self.army_generator
        )
        self.history = [self.state.copy()]
        return [self.state.GetPlayerState(i) for i in range(self.num_players)]

class GeneralsSingleAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(
        self,
        map0 = None,
        Ws = [5],
        Hs = [3],
        enemy_agents = [],
        p_mountain = 0.2,
        p_city = 0.05,
        army_generator = UniformArmyGenerator(25,101)
    ):
        super(GeneralsEnv, self).__init__()
        self.map = map0; self.state = None
        self.Ws = Ws; self.Hs = Hs
        self.num_players = len(enemy_agents)+1
        self.enemy_agents = enemy_agents
        self.p_mountain = p_mountain
        self.p_city = p_city
        self.army_generator = army_generator
        self.history = []
    
    def step(self, action):
        old_observations = [self.state.GetPlayerState(i) for i in range(self.num_players)]
        actions = [action] + [agent.take_action(old_observations[i]) for i in range(1,self.num_players)]
        done = self.state.GetNextState_(actions); self.history.append(self.state.copy())
        new_observation = new_state.GetPlayerState(0)
        reward = new_observation.Score() - new_observation.Score()
        info = {}
        return new_observation, reward, done, info

    def reset(self):
        self.state = NewBoardStateFromMap(
            NewRandomMap(
                np.random.choice(self.Ws),
                self.num_players,
                np.random.choice(self.Hs),
                self.p_mountain,
                self.p_city,
            )
            if self.map is None else self.map,
            army_generator = self.army_generator
        )
        self.history = [self.state.copy()]
        return self.state.GetPlayerState(0)
