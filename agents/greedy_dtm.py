from .base import *
from policy.omniscienceSearch import greedyActions

class DeterminedGreedyAgent(BaseAgent):
    def get_action(self, obs, **info):
        return greedyActions(obs.board, self.agent_id)
