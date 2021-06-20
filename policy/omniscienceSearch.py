import tianshou as ts
from tianshou.policy import BasePolicy
from tianshou.data import Batch
import torch
from torch.distributions import Categorical
from env.states import BoardState, PlayerAction
from torch.nn.functional import softmax
import numpy as np

class AlphaBetaSearch:
    iter = 0
    max_actions = 3
    @classmethod
    def maxValue(cls, alpha, beta, currentDepth, state,agent_id, depth, truncated=True):
        cls.iter +=1
        if cls.iter%10000 ==0:
            print(cls.iter)
        if currentDepth == depth:
            return state.GetPlayerState(agent_id).Score() - state.GetPlayerState(1-agent_id).Score()
        value = -1e9
        if truncated:
            act_list = sampledGreedyActions(state,agent_id,number=cls.max_actions,serialize=False)
        else:
            player_state = state.GetPlayerState(agent_id)
            act_list = player_state.AvailableActions(serialize=False)
            act_list.append(PlayerAction((0, 0), (-1, 0), False))

        for act in act_list:
            move = [None] * state.num_players
            move[agent_id] = act
            new_state, end = state.GetNextState(moves=move)
            if end:
                value = new_state.GetPlayerState(agent_id).Score()-state.GetPlayerState(1-agent_id).Score()
            else:
                value = -cls.maxValue(-beta, -alpha, currentDepth+1, new_state, 1 - agent_id,depth)
            if value >= beta:
                break
            alpha = max(alpha, value)
        return value

    @classmethod
    def minmaxAction(cls, state:BoardState, agent_id, depth, truncated=True):
        best = -1e9
        best_action = 0
        player_state = state.GetPlayerState(agent_id)
        act_list = player_state.AvailableActions(serialize=False)
        for act in act_list:
            move = [None]*state.num_players
            move[agent_id] = act
            new_state, end = state.GetNextState(moves=move)
            if end:
                value = new_state.GetPlayerState(agent_id).Score()-new_state.GetPlayerState(1-agent_id).Score()
            else:
                value = -cls.maxValue(-1e9, -best, 0, new_state,1-agent_id, depth,truncated)
            if value > best:
                best = value
                best_action = act.serializein(state)
        return best_action

    @classmethod
    def sampledMinmaxAction(cls, state:BoardState, agent_id, depth, truncated=True, beta = 10):
        act_value = []
        best = -1e9
        player_state = state.GetPlayerState(agent_id)
        act_list = player_state.AvailableActions(serialize=False)
        act_list.append(PlayerAction((0,0),(-1,0),False))
        for act in act_list:
            move = [None] * state.num_players
            move[agent_id] = act
            new_state, end = state.GetNextState(moves=move)
            if end:
                value = new_state.GetPlayerState(agent_id).Score() - new_state.GetPlayerState(1-agent_id).Score()
            else:
                value = -cls.maxValue(-1e9, -best, 0, new_state, 1-agent_id, depth,truncated)
            act_value.append(value)
            best = max(best, value)
        prob = softmax(torch.tensor(act_value) * beta, dim=0).tolist()
        select = np.random.choice(a=len(act_list), size=1, replace=False, p=prob).item()
        act = act_list[select].serializein(state)
        return act


def greedyActions(state, agent_index):
    player_state = state.GetPlayerState(agent_index)
    pre_score = player_state.Score()
    act_lists = player_state.AvailableActions(serialize=False)
    best_reward = -1e10
    best_act = 0
    for act in act_lists:
        move = [None] * state.num_players
        move[agent_index] = act
        next_state, _ = state.GetNextState(moves=move)
        next_player_state = next_state.GetPlayerState(agent_index)
        score = next_player_state.Score()
        if score - pre_score > best_reward:
            best_act = act.serializein(state)
            best_reward = score - pre_score
        #print(f"best:{best_reward}")
        #print(f"score: {score, pre_score}")
    return best_act


def sampledGreedyActions(state, agent_index, number = 1, serialize = True, beta = 10):
    player_state = state.GetPlayerState(agent_index)
    pre_score = player_state.Score()
    act_lists = player_state.AvailableActions(serialize=False)
    act_lists.append(PlayerAction((0,0),(-1,0),False))
    reward = []
    for act in act_lists:
        move = [None] * state.num_players
        move[agent_index] = act
        next_state, _ = state.GetNextState(moves=move)
        next_player_state = next_state.GetPlayerState(agent_index)
        score = next_player_state.Score()
        reward.append(score - pre_score)
        # print(f"best:{best_reward}")
        # print(f"score: {score, pre_score}")
    prob = softmax(torch.tensor(reward) * beta,dim = 0).tolist()
    select = np.random.choice(a = len(act_lists),size = min(number,len(act_lists)),replace = False,p = prob)
    acts = np.array(act_lists)[select].tolist()
    return acts[0].serializein(state) if serialize else acts


class OmniscienceSearch(BasePolicy):
    def __init__(self,  sampled = True, depth = 2):
        super(OmniscienceSearch, self).__init__()
        self.sampled = sampled
        self.depth = depth

    def set_eps(self, rate):
        pass

    def forward(
            self,
            batch: Batch,
            state=None,
            **kwargs,
    ) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which MUST have the following keys:

            * ``act`` an numpy.ndarray or a torch.Tensor, the action over \
                given batch data.
            * ``state`` a dict, an numpy.ndarray or a torch.Tensor, the \
                internal state of the policy, ``None`` as default.
        """
        agent_index = self.agent_id - 1
        acts = []
        for board in batch.board:
            if self.depth is None:
                if self.sampled:
                    act = sampledGreedyActions(board, agent_index)
                else:
                    act = greedyActions(board,agent_index)
            else:
                if self.sampled:
                    act = AlphaBetaSearch.sampledMinmaxAction(board,agent_index,self.depth)
                else:
                    act = AlphaBetaSearch.minmaxAction(board,agent_index,self.depth)
            acts.append(act)
        return Batch(act = acts,logits= None, state = None)

    def learn(self, batch: Batch, **kwargs):
        """Update policy with a given batch of data.

        :return: A dict, including the data needed to be logged (e.g., loss).

        .. note::

            In order to distinguish the collecting state, updating state and
            testing state, you can check the policy state by ``self.training``
            and ``self.updating``. Please refer to :ref:`policy_state` for more
            detailed explanation.

        .. warning::

            If you use ``torch.distributions.Normal`` and
            ``torch.distributions.Categorical`` to calculate the log_prob,
            please be careful about the shape: Categorical distribution gives
            "[batch_size]" shape while Normal distribution gives "[batch_size,
            1]" shape. The auto-broadcasting of numerical operation with torch
            tensors will amplify this error.
        """
        return {'loss':0.0}
