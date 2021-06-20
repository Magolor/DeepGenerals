import env
import agents
from tianshou.data import Batch

def get_agent_batch(batch, agent_id):
    obs = batch.obs[agent_id].unsqueeze(0) if hasattr(batch, 'obs') else None
    board = batch.board[agent_id] if hasattr(batch,'board') else None
    net_batch = Batch({'obs': obs, 'board': board})
    return net_batch

def play(agents, name="Default", auto_replay_id=True):
    game = env.env("generals",name=name, auto_replay_id=auto_replay_id); S, r, d, i = game.reset(), 0, False, None
    for agent_id, agent in enumerate(agents):
        agent.reset(agent_id=agent_id,obs=get_agent_batch(S,agent_id))
    c = 0
    while not d:
        print(c); c += 1
        actions = [agent.get_action(obs=get_agent_batch(S,agent_id),info=i) for agent_id, agent in enumerate(agents)]
        S, r, d, i = game.step(actions)
        # input()

if __name__=="__main__":
    play([agents.OmniAgent(),agents.HumanAgent(cheat=False)], name="PlayerFrameworkTesting")