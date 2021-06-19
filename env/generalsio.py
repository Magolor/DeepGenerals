import sys
import gym
import pickle
sys.path.append("./")
from utils import *
from env.states import *

class GeneralsMultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(
        self,
        map0 = None,
        Ws = [8,7,6],
        Hs = [8,7,6],
        num_players = 3,
        p_mountain = 0.2,
        p_city = 0.04,
        army_generator = UniformArmyGenerator(25,101),
        **kwargs
    ):
        super(GeneralsMultiAgentEnv, self).__init__()
        self.map = map0; self.state = None
        self.Ws = Ws; self.Hs = Hs
        self.Wmax = max(Ws); self.Hmax = max(Hs)
        self.num_players = num_players
        self.p_mountain = p_mountain
        self.p_city = p_city
        self.army_generator = army_generator
        self.history = []
        self.step_result = None
        self.replay_id = None
    
    def step(self, actions):
        old_observations = [[h.GetPlayerState(i) for h in self.history[-C.NUM_FRAME:]] for i in range(self.num_players)]
        done = self.state.GetNextState_(actions); self.history.append(self.state.copy()); force_done = (len(self.history)>C.NUM_FRAME+C.MAX_TURN) and (not done)
        new_observations = [[h.GetPlayerState(i) for h in self.history[-C.NUM_FRAME:]] for i in range(self.num_players)]
        rewards = [
        (   new_observations[i][-1].Score()
        -   old_observations[i][-1].Score()
        -   C.TIME_PUNISHMENT * C.REWARD_SCALE
        -   force_done * 0.1 * C.REWARD_SCALE)
        for i in range(self.num_players)
        ]
        done = done or force_done
        self.step_result = (new_observations, rewards, done, {'god':[self.state.GetPlayerState(i) for i in range(self.num_players)],
                                                              'board':[self.state]*self.state.num_players})
        if done:
            self.save_replay()
    
    def last(self):
        return self.step_result

    # def step(self, action):
    #     self.actions.append(action); self.agent_selection = (self.agent_selection+1)%self.num_players
    #     while self.agent_selection in self.state.dead:
    #         self.actions.append(None); self.agent_selection = (self.agent_selection+1)%self.num_players
    #     if len(self.actions) == self.num_players:
    #         old_observations = [self.state.GetPlayerState(i) for i in range(self.num_players)]
    #         done = self.state.GetNextState_(self.actions); self.history.append(self.state.copy())
    #         new_observations = [new_state.GetPlayerState(i) for i in range(self.num_players)]
    #         rewards = [new_observations[i].Score() - old_observations[i].Score() for i in range(self.num_players)]
    #         info = {}

    def reset(self, replay_id=None):
        if self.map is None:
            map0 = NewRandomMap(
                np.random.choice(self.Ws),
                np.random.choice(self.Hs),
                self.num_players,
                self.p_mountain,
                self.p_city,
            )
            map0.pad_to(self.Wmax, self.Hmax)
            self.state = NewBoardStateFromMap(
                map0, army_generator = self.army_generator
            )
        else:
            self.state = NewBoardStateFromMap(
                self.map, army_generator = self.army_generator
            )
        self.history = [self.state.copy() for _ in range(C.NUM_FRAME)]
        self.step_result = ([[h.GetPlayerState(i) for h in self.history[-C.NUM_FRAME:]] for i in range(self.num_players)], None, None,
                            {'god':[self.state.GetPlayerState(i) for i in range(self.num_players)],
                             'board':[self.state]*self.state.num_players})
        self.replay_id = replay_id
        return self.step_result

    def save_replay(self):
        if self.replay_id is not None:
            pickle.dump([h.serialize() for h in self.history],open("replays/"+self.replay_id+".replay","wb"))

# class GeneralsSingleAgentEnv(gym.Env):
#     metadata = {
#         'render.modes': ['human', 'rgb_array'],
#         'video.frames_per_second': 2
#     }
# 
#     def __init__(
#         self,
#         map0 = None,
#         Ws = [5],
#         Hs = [3],
#         enemy_agents = [],
#         p_mountain = 0.2,
#         p_city = 0.05,
#         army_generator = UniformArmyGenerator(25,101)
#     ):
#         super(GeneralsSingleAgentEnv, self).__init__()
#         self.map = map0; self.state = None
#         self.Ws = Ws; self.Hs = Hs
#         self.num_players = len(enemy_agents)+1
#         self.enemy_agents = enemy_agents
#         self.p_mountain = p_mountain
#         self.p_city = p_city
#         self.army_generator = army_generator
#         self.history = []
#     
#     def step(self, action):
#         old_observations = [self.state.GetPlayerState(i) for i in range(self.num_players)]
#         actions = [action] + [agent.take_action(old_observations[i]) for i in range(1,self.num_players)]
#         done = self.state.GetNextState_(actions); self.history.append(self.state.copy())
#         new_observation = new_state.GetPlayerState(0)
#         reward = new_observation.Score() - new_observation.Score()
#         info = {}
#         return new_observation, reward, done, info
# 
#     def reset(self):
#         self.state = NewBoardStateFromMap(
#             NewRandomMap(
#                 np.random.choice(self.Ws),
#                 self.num_players,
#                 np.random.choice(self.Hs),
#                 self.p_mountain,
#                 self.p_city,
#             )
#             if self.map is None else self.map,
#             army_generator = self.army_generator
#         )
#         self.history = [self.state.copy()]
#         return self.state.GetPlayerState(0)
# 
# 