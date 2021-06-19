import env
import agents
from tianshou.data import Batch

def get_agent_batch(batch, agent_id):
    obs = batch.obs if not hasattr(batch.obs, 'obs') else batch.obs.obs[:, agent_id]
    board = batch.obs.board if hasattr(batch.obs,'board') else None
    net_batch = Batch({'obs': obs, 'board': board})
    return net_batch

def play(agents, name="Default", auto_replay_id=True):
    game = env.env("generals",name=name, auto_replay_id=auto_replay_id); S, r, d, i = game.reset(), 0, False, None
    for agent_id, agent in enumerate(agents):
        agent.reset(agent_id=agent_id)
    while not d:
        actions = [agent.get_action(obs=get_agent_batch(S,agent_id),board_state=S.board,info=i) for agent_id, agent in enumerate(agents)]
        S, r, d, i = game.step(actions)

if __name__=="__main__":
    play([agents.BaseAgent(),agents.BaseAgent()], name="PlayerFrameworkTesting")