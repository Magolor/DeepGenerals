import env
import agents
from tianshou.data import Batch

def get_agent_batch(batch, agent_id):
    obs = batch.obs[agent_id].unsqueeze(0) if hasattr(batch, 'obs') else None
    board = batch.board[agent_id] if hasattr(batch,'board') else None
    net_batch = Batch({'obs': obs, 'board': board}); print(board.turn)
    return net_batch

def play(agents, name="Default", auto_replay_id=True):
    game = env.env("generals",name=name, auto_replay_id=auto_replay_id); S, r, d, i = game.reset(), 0, False, None
    for agent_id, agent in enumerate(agents):
        agent.reset(agent_id=agent_id,obs=get_agent_batch(S,agent_id))
    while not d:
        actions = [agent.get_action(obs=get_agent_batch(S,agent_id),info=i) for agent_id, agent in enumerate(agents)]
        S, r, d, i = game.step(actions)

    for agent_id, result in enumerate([get_agent_batch(S,agent_id) for agent_id in range(len(agents))]):
        if agent_id not in result.board.dead:
            print("Player %d wins!"%(agent_id+1))

if __name__=="__main__":
    play([agents.RandomGreedyAgent(),agents.HumanAgent(cheat=True)], name="PlayerFrameworkTesting")