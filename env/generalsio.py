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
        Ws = [6], # [16,15,14,13,12],
        Hs = [5], # [16,15,14,13,12],
        num_players = 2,
        p_mountain = 0.05,
        p_city = 0.10,
        army_generator = UniformArmyGenerator(40,61),
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
        (   (new_observations[i][-1].Score())
        -   (old_observations[i][-1].Score())
        +   (actions[i].IsAvailableIn(old_observations[i][-1]) * 1 if actions[i] is not None else 0)
        +   (actions[i].IsEffectiveIn(old_observations[i][-1]) * 1 if actions[i] is not None else 0)
        +   (actions[i].IsOffensiveIn(old_observations[i][-1]) * 1 if actions[i] is not None else 0)
        -   (C.TIME_PUNISHMENT * C.REWARD_SCALE)
        -   (force_done * 0.1 * C.REWARD_SCALE))
        for i in range(self.num_players)
        ]
        print(
            "***",
            rewards[0],
            new_observations[0][-1].Score(),
        -   old_observations[0][-1].Score(),
        +   actions[0].IsAvailableIn(old_observations[0][-1]) * 1 if actions[0] is not None else 0,
        +   actions[0].IsEffectiveIn(old_observations[0][-1]) * 1 if actions[0] is not None else 0,
        +   actions[0].IsOffensiveIn(old_observations[0][-1]) * 1 if actions[0] is not None else 0,
        -   C.TIME_PUNISHMENT * C.REWARD_SCALE,
        -   force_done * 0.1 * C.REWARD_SCALE,
        )
        done = done or force_done
        self.step_result = (new_observations, rewards, done, {'god':[self.state.GetPlayerState(i) for i in range(self.num_players)],
                                                              'board':[self.state]*self.state.num_players})
        if done:
            self.save_replay()
    
    def last(self):
        return self.step_result

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
