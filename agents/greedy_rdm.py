from .base import *
from policy.omniscienceSearch import sampledGreedyActions

class RandomGreedyAgent(BaseAgent):
    def get_action(self, obs, **info):
        return sampledGreedyActions(obs.board, self.agent_id)
