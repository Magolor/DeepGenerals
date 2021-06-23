from .base import *
from policy.omniscienceSearch import greedyActions,sampledActions

class RandomAgent(BaseAgent):
    def get_action(self, obs, **info):
        return sampledActions(obs.board, self.agent_id)
