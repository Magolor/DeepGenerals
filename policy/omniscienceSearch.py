import tianshou as ts
from tianshou.policy import BasePolicy
from tianshou.data import Batch
import torch
from torch.distributions import Categorical
from env.states import BoardState, PlayerAction
from torch.nn.functional import softmax
import numpy as np
from tool import Timer
from env.const import C

class AlphaBetaSearch:
    iter = 0
    max_actions = 1
    timer = Timer()
    @classmethod
    def maxValue(cls, currentDepth, state,agent_id, depth, truncated=True):
        cls.iter +=1
        print(cls.iter)
        if cls.iter%10000 ==0:
            print(cls.iter)
        if currentDepth == depth:
            return state.GetPlayerState(agent_id).Score()
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
            new_state, end = state.GetNextState(actions=move)
            if end:
                value = new_state.GetPlayerState(agent_id).Score()
            else:
                value = -cls.randomValue(currentDepth+1, new_state, 1 - agent_id,depth)

        return value

    @classmethod
    def randomValue(cls, currentDepth, state, agent_id, depth, truncated=True):
        if currentDepth == depth:
            return -state.GetPlayerState(1-agent_id).Score()
        player_state = state.GetPlayerState(agent_id)
        act_list = player_state.AvailableActions(serialize=False)
        act_list.append(PlayerAction((0, 0), (-1, 0), False))

        act = act_list[np.random.randint(len(act_list))]
        move = [None] * state.num_players
        move[agent_id] = act
        new_state, end = state.GetNextState(actions=move)
        if end:
            value = -new_state.GetPlayerState(1-agent_id).Score()
        else:
            value = -cls.maxValue(currentDepth + 1, new_state, 1 - agent_id, depth)
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
            new_state, end = state.GetNextState(actions=move)
            if end:
                value = new_state.GetPlayerState(agent_id).Score()
            else:
                value = -cls.maxValue(-1e9, -best, 0, new_state,1-agent_id, depth,truncated)
            if value > best:
                best = value
                best_action = act.serializein(state)
        return best_action

    @classmethod
    def sampledMinmaxAction(cls, state:BoardState, agent_id, depth, truncated=True, beta = 5):
        act_value = []
        player_state = state.GetPlayerState(agent_id)
        act_list = player_state.AvailableActions(serialize=False)
        for act in act_list:
            move = [None] * state.num_players
            move[agent_id] = act
            new_state, end = state.GetNextState(actions=move)
            if end:
                value = new_state.GetPlayerState(agent_id).Score()
            else:
                value = -cls.randomValue(0, new_state, 1-agent_id, depth,truncated)
            act_value.append(value)
        act_value.append(0.0)
        act_list.append(PlayerAction((0,0),(-1,0),False))
        prob = softmax(torch.tensor(act_value) * beta, dim=0).tolist()
        select = np.random.choice(a=len(act_list), size=1, replace=False, p=prob).item()
        act = act_list[select].serializein(state)
        return act

    @classmethod
    def sampledAlternativeAction(cls, state: BoardState, agent_id, depth, beta = 5):
        act_value = []
        act_list = state.AvailableActions(serialize=False,player_id=agent_id)
        for act in act_list:
            move = [None] * state.num_players
            move[agent_id] = act
            new_state, end = state.GetNextState(actions=move,update_observation=False)
            #print(depth, end)
            for counter in range(depth * (1-end)):
                #print(counter)
                new_move = [None] * new_state.num_players
                c = agent_id if counter&1 else 1-agent_id
                new_move[c] = sampledGreedyActions(new_state,c,1,False,beta)[0]
                '''
                if counter % 2 == 0:
                    new_move[1-agent_id] = sampledActions(new_state,1-agent_id,1,False)
                else:
                    new_move[agent_id] = sampledGreedyActions(new_state,agent_id,1,False,beta=beta)
                '''
                end = new_state.GetNextState_(actions=new_move,update_observation=False)
                if end:
                    break
            value = new_state.Score(player_id=agent_id) \
                    + act.IsEffectiveIn(state) * C.ACTION_REWARD \
                    + act.IsOffensiveIn(state) * C.ACTION_REWARD
            act_value.append(value)

        act_value.append(0.0)
        act_list.append(PlayerAction((0, 0), (-1, 0), False))
        prob = softmax(torch.tensor(act_value) * beta, dim=0).tolist()
        select = np.random.choice(a=len(act_list), size=1, replace=False, p=prob).item()
        act = act_list[select].serializein(state)
        return act


def sampledActions(state, agent_index, number = 1, serialize = True):
    act_lists = state.AvailableActions(serialize=False,player_id=agent_index)
    act_lists.append(PlayerAction((0, 0), (-1, 0), False))
    select = np.random.choice(a = len(act_lists),size = min(number,len(act_lists)),replace = False)
    acts = np.array(act_lists)[select].tolist()
    return acts[0].serializein(state) if serialize else acts

def greedyActions(state, agent_index):
    act_lists = state.AvailableActions(serialize=False,player_id=agent_index)
    best_reward = -1e10
    best_act = 0
    for act in act_lists:
        move = [None] * state.num_players
        move[agent_index] = act
        next_state, _ = state.GetNextState(actions=move,update_observation=False)
        score = next_state.Score(player_id=agent_index)
        reward = score \
                 + act.IsEffectiveIn(state) * C.ACTION_REWARD \
                 + act.IsOffensiveIn(state) * C.ACTION_REWARD

        if reward > best_reward:
            best_act = act.serializein(state)
            best_reward = reward
        #print(f"best:{best_reward}")
        #print(f"score: {score, pre_score}")
    return best_act


def sampledGreedyActions(state, agent_index, number = 1, serialize = True, beta = 5):
    act_lists = state.AvailableActions(serialize=False,player_id=agent_index)
    reward = []
    for act in act_lists:
        move = [None] * state.num_players
        move[agent_index] = act
        next_state, _ = state.GetNextState(actions=move,update_observation=False)
        score = next_state.Score(player_id=agent_index)
        reward_ = score \
                 + act.IsEffectiveIn(state) * C.ACTION_REWARD \
                 + act.IsOffensiveIn(state) * C.ACTION_REWARD
        reward.append(reward_)
        # print(f"best:{best_reward}")
        # print(f"score: {score, pre_score}")
    reward.append(0.0)
    act_lists.append(PlayerAction((0, 0), (-1, 0), False))
    prob = softmax(torch.tensor(reward) * beta,dim = 0).tolist()
    select = np.random.choice(a = len(act_lists),size = min(number,len(act_lists)), replace = False, p = prob)
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
