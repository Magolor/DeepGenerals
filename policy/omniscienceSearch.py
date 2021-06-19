import tianshou as ts
from tianshou.policy import BasePolicy
from tianshou.data import Batch
import numpy as np


class OmniscienceSearch(BasePolicy):
    def __init__(self):
        super(OmniscienceSearch, self).__init__()

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
        for board, playerState in zip(batch.board,batch.god):
            pre_score = playerState.Score()
            act_lists = playerState.AvailableActions(serialize=False)
            best_act = None
            best_reward = 0
            for act in act_lists:
                move = [None]* board.num_players
                move[agent_index] = act
                state, _ = board.GetNextState(moves=move)
                nextState = state.GetPlayerState(self, agent_index)
                score = nextState.Score()
                if score-pre_score>best_reward:
                    best_act = act.serializein(board)
                    best_reward = score-pre_score
            acts.append(best_act)
        return Batch(act = acts,logits= None)

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
        pass
