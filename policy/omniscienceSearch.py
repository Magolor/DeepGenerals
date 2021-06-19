import tianshou as ts
from tianshou.policy import BasePolicy
from tianshou.data import Batch

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


class OmniscienceSearch(BasePolicy):
    def __init__(self, step = 1):
        super(OmniscienceSearch, self).__init__()

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
            act = greedyActions(board,agent_index)
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
