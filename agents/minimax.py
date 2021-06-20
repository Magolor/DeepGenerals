from .base import *
from policy.omniscienceSearch import AlphaBetaSearch

class MinimaxAgent(BaseAgent):
    def get_action(self, obs, **info):
        return AlphaBetaSearch.sampledMinmaxAction(obs.board, self.agent_id, depth=2, truncated=True)
