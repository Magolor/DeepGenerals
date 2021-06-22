from .base import *
from policy.omniscienceSearch import AlphaBetaSearch

class MinimaxAgent(BaseAgent):
    def get_action(self, obs, **info):
        return AlphaBetaSearch.sampledAlternativeAction(obs.board, self.agent_id, depth=3)
