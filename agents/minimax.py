from .base import *
from policy.omniscienceSearch import AlphaBetaSearch

class MinimaxAgent(BaseAgent):
    def __init__(self, **kwargs):
        self.beta = kwargs.pop('beta')
        self.depth = kwargs.pop('depth')
        super(MinimaxAgent, self).__init__()

    def get_action(self, obs, **info):
        return AlphaBetaSearch.sampledAlternativeAction(obs.board, self.agent_id, depth=self.depth, beta=self.beta)
