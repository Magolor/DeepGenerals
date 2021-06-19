import env

class BaseAgent(object):
    def __init__(self, **kwargs):
        self.agent_id = None

    def reset(self, agent_id):
        self.agent_id = agent_id

    def get_action(self, obs, board_state=None, **info):
        # return PlayerAction type or None
        return None