import env
import agents
from tianshou.data import Batch
from utils import *

AGENT_POOL = [
    # agents.BaseAgent(),
    agents.DeterminedGreedyAgent(beta=5),
    agents.RandomGreedyAgent(beta=5),
]
def NAME(agent):
    return agent.__class__.__name__

def get_agent_batch(batch, agent_id):
    obs = batch.obs[agent_id].unsqueeze(0) if hasattr(batch, 'obs') else None
    board = batch.board[agent_id] if hasattr(batch,'board') else None
    net_batch = Batch({'obs': obs, 'board': board})
    return net_batch

def play(agents, name="Default", auto_replay_id=True, print_step=True):
    game = env.env("generals",name=name, auto_replay_id=auto_replay_id); S, r, d, i = game.reset(), 0, False, None
    for agent_id, agent in enumerate(agents):
        agent.reset(agent_id=agent_id,obs=get_agent_batch(S,agent_id))
    c = 0
    if print_step:
        print(HIGHLIGHT("%04d"%c), end=' ', flush=True)
    while not d:
        actions = [agent.get_action(obs=get_agent_batch(S,agent_id),info=i) for agent_id, agent in enumerate(agents)]
        S, r, d, i = game.step(actions)
        if print_step:
            print("\b"*5+"%s"%HIGHLIGHT("%04d"%c), end=' ', flush=True); c += 1

    for agent_id, result in enumerate([get_agent_batch(S,agent_id) for agent_id in range(len(agents))]):
        if len(result.board.dead) != len(agents)-1:
            if print_step:
                print("\b"*5+"%s"%WARN("%04d"%c), end=' ', flush=True)
            return -1
        if agent_id not in result.board.dead:
            if agent_id==0 and print_step:
                print("\b"*5+"%s"%SUCCESS("%04d"%c), end=' ', flush=True)
            else:
                print("\b"*5+"%s"%ERROR("%04d"%c), end=' ', flush=True)
            return agent_id

def evaluate(agent, round=23):
    winrates = {}
    for enemy in AGENT_POOL:
        print("Evaluating [%s] v.s. [%s]..."%(HIGHLIGHT(NAME(agent)),HIGHLIGHT(NAME(enemy))), end = ' ', flush=True)
        winners = [play([agent, enemy], name = "Evaluate[%s][%s]"%(NAME(agent),NAME(enemy))) for _ in range(round)]
        winrates[NAME(enemy)] = sum([winner==0 for winner in winners])/len(winners); print(SUCCESS("Done."))
    return winrates

def evaluate_neural(pt_folder, round=23, start=1, device='cpu'):
    root = os.path.join(pt_folder,"checkpoint"); agent_pts = sorted(os.listdir(root)); name = "NeuralAgent"
    metrics = [NAME(agent) for agent in AGENT_POOL]; registrations = [('epoch',name,'greater') for name in metrics]
    with Tracker(title=name,DIR=os.path.join(pt_folder,"evaluate"),registrations=registrations) as T:
        for epoch, agent_pt in TQDM(agent_pts[start-1:],s=start):
            #assert(agent_pt=="Epoch%05d.pt"%epoch);
            agent = agents.NeuralAgent(path=os.path.join(root,agent_pt),device=device)
            for key, value in evaluate(agent, round):
                T.update(key, value, epoch)

if __name__=="__main__":
    # exp_name = "DG_Big"
    # evaluate_neural(os.path.join("Experiment",exp_name),start=15)
    # other = agents.MinimaxAgent()
    # human = agents.HumanAgent(cheat = True)
    play([agents.RandomGreedyAgent(beta=5),agents.RandomGreedyAgent(beta=5)])
