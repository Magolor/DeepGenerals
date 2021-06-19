from env.states import PlayerActionFromID

class BaseAgent(object):
    def __init__(self, **kwargs):
        self.agent_id = None

    def reset(self, agent_id):
        self.agent_id = agent_id

    def get_action(self, obs, **info):
        # return int type or None
        return None