from .base import *
from policy.omniscienceSearch import sampledGreedyActions

class RandomGreedyAgent(BaseAgent):
    def __init__(self, **kwargs):
        self.beta = kwargs.pop('beta')
        super(RandomGreedyAgent, self).__init__()

    def get_action(self, obs, **info):
        return sampledGreedyActions(obs.board, self.agent_id, beta=self.beta)
