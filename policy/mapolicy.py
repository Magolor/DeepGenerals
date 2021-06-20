import numpy as np
import torch
from typing import Any, Dict, List, Tuple, Union, Optional

from tianshou.policy import BasePolicy
from tianshou.data import Batch, ReplayBuffer, VectorReplayBuffer
from copy import deepcopy
from policy.omniscienceSearch import OmniscienceSearch, sampledGreedyActions

#class SMAReplayBuffer(ReplayBuffer):
#	def __init__(self, agent_id):
#       super(SMAReplayBuffer, self).__init__()
#		self.agent_id = agent_id

#	def __item__(self, index):
#		if isinstance(index, int):
#			return self.buffer[index][self.agent_id]


class MultiAgentPolicyManager(BasePolicy):
    """Multi-agent policy manager for MARL.

    This multi-agent policy manager accepts a list of
    :class:`~tianshou.policy.BasePolicy`. It dispatches the batch data to each
    of these policies when the "forward" is called. The same as "process_fn"
    and "learn": it splits the data and feeds them to each policy. A figure in
    :ref:`marl_example` can help you better understand this procedure.
    """

    def __init__(self, policies: List[BasePolicy], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.policies = policies
        for i, policy in enumerate(policies):
            # agent_id 0 is reserved for the environment proxy
            # (this MultiAgentPolicyManager)
            policy.set_agent_id(i + 1)
            if hasattr(policy, 'set_eps'):
                policy.set_eps(0)
        self.rate = 0

    def set_eps(self, rate):
        self.rate = rate

    def replace_policy(self, policy: BasePolicy, agent_id: int) -> None:
        """Replace the "agent_id"th policy in this manager."""
        self.policies[agent_id - 1] = policy
        policy.set_agent_id(agent_id)

    def adapt_buffer(self, buffer, indice, agent_index):
        new_buffer =  deepcopy(buffer)
        new_buffer._meta['obs'].obs = buffer.obs.obs[:, agent_index]
        new_buffer._meta['act'] =    buffer.act[:, agent_index]
        new_buffer._meta['rew'] =    buffer.rew[:, agent_index]
        new_buffer._meta['obs_next'].obs = buffer.obs_next.obs[:, agent_index]
        return new_buffer


    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        """Dispatch batch data from obs.agent_id to every policy's process_fn.

        Save original multi-dimensional rew in "save_rew", set rew to the
        reward of each agent during their "process_fn", and restore the
        original reward afterwards.
        """
        results = {}
        for policy in self.policies:
            agent_index = policy.agent_id-1

            tmp_batch,_ = self.get_agent_batch(batch, agent_index)
            tmp_indice = indice
            results[f"agent_{policy.agent_id}"] = policy.process_fn(
                tmp_batch, self.adapt_buffer(buffer, indice, agent_index), tmp_indice)
        return Batch(results)

    def get_agent_batch(self, batch, agent_id):
        obs = batch.obs if not hasattr(batch.obs, 'obs') else batch.obs.obs[:, agent_id]
        obs_next = batch.obs_next if not hasattr(batch.obs_next, 'obs') else batch.obs_next.obs[:, agent_id]
        #if self.act is
        act = batch.act[:, agent_id] if isinstance(batch.act, np.ndarray) else batch.act
        rew = batch.rew[:, agent_id] if isinstance(batch.rew, np.ndarray) else batch.rew
        god = None if not hasattr(batch.obs, 'god') else batch.obs.god[:, agent_id]
        board = batch.obs.board[:,agent_id] if hasattr(batch.obs,'board') else None
        net_batch = Batch({
            'obs': obs,
            'done': batch.done,
            'info': batch.info,
            'policy': batch.policy,
            'rew': rew,
            'act': act,
            'obs_next': obs_next,
            'board': board
        })
        return net_batch, god

    def exploration_noise(
        self, act: Union[np.ndarray, Batch], batch: Batch
    ) -> Union[np.ndarray, Batch]:
        """Add exploration noise from sub-policy onto act."""
        for policy in self.policies:
            # change to agent
            act[0][policy.agent_id-1] = policy.exploration_noise(
                act[0][policy.agent_id-1], batch)
        return act

    def forward(  # type: ignore
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Dispatch batch data from obs.agent_id to every policy's forward.

        :param state: if None, it means all agents have no state. If not
            None, it should contain keys of "agent_1", "agent_2", ...

        :return: a Batch with the following contents:

        ::

            {
                "act": actions corresponding to the input
                "state": {
                    "agent_1": output state of agent_1's policy for the state
                    "agent_2": xxx
                    ...
                    "agent_n": xxx}
                "out": {
                    "agent_1": output of agent_1's policy for the input
                    "agent_2": xxx
                    ...
                    "agent_n": xxx}
            }
        """
        results: List[Tuple[bool, np.ndarray, Batch,
                            Union[np.ndarray, Batch], Batch]] = []
        #assert len(batch) == 1
        for policy in self.policies:
            # This part of code is difficult to understand.
            # Let's follow an example with two agents
            # batch.obs.agent_id is [1, 2, 1, 2, 1, 2] (with batch_size == 6)
            # each agent plays for three transitions
            # agent_index for agent 1 is [0, 2, 4]
            # agent_index for agent 2 is [1, 3, 5]
            # we separate the transition of each agent according to agent_id

            agent_index = policy.agent_id -1
            tmp_batch, god = self.get_agent_batch(batch,agent_index)

            # compute action
            acts = []
            logits = []
            out = None

            # check if it is network policy
            if isinstance(policy, OmniscienceSearch):
                out = policy(batch=tmp_batch, state=None if state is None
                else state["agent_" + str(policy.agent_id)],
                             **kwargs)
            else:
                if np.random.rand() > self.rate:
                    raw_out = policy(batch=tmp_batch, state=None if state is None
                    else state["agent_" + str(policy.agent_id)],
                                 **kwargs)
                    # mask-out infeasible actions
                    mask = []
                    for g in god:
                        act_lists = g.AvailableActions(serialize = True)
                        mask.append([(i in act_lists) for i in range(np.prod(g.board_shape)*8)])
                    logits = raw_out.logits.cpu() * torch.tensor(mask,dtype=torch.float)
                    acts = torch.argmax(logits,dim = 1).cpu().numpy()
                    out = Batch(act = acts,
                                logtis = logits,
                                state = None)
                # explore
                else:
                    if np.random.rand()>1:
                        for b, g in enumerate(god):
                            act_lists = g.AvailableActions(serialize=True)
                            act_lists.append(0)
                            act = np.random.choice(act_lists)
                            logit = torch.zeros(size = (np.prod(g.board_shape)*8,), dtype = torch.float)
                            logit[act] = 1
                            acts.append(act)
                            logits.append(logit)
                            out = Batch(act = np.array(acts),
                                        logits = torch.stack(logits,dim=0),
                                        state = None)
                    else:
                        for board, g in zip(tmp_batch.board, god):
                            act_lists = g.AvailableActions(serialize=True)
                            act_lists.append(0)
                            act = sampledGreedyActions(board,agent_index,beta=2)
                            logit = torch.zeros(size=(np.prod(g.board_shape) * 8,), dtype=torch.float)
                            logit[act] = 1
                            acts.append(act)
                            logits.append(logit)
                            out = Batch(act=np.array(acts),
                                        logits=torch.stack(logits, dim=0),
                                        state=None)
            act = out.act
            each_state = out.state \
                if (hasattr(out, "state") and out.state is not None) \
                else Batch()
            results.append((True, agent_index, out, act, each_state))
        act = np.array([result[3] for result in results]).transpose()
        holder = Batch({'act': act})
        state_dict, out_dict = {}, {}
        for policy, (has_data, agent_index, out, act, state) in zip(
                self.policies, results):
            state_dict["agent_" + str(policy.agent_id)] = state
            out_dict["agent_" + str(policy.agent_id)] = out
        holder["out"] = out_dict
        holder["state"] = state_dict
        return holder

    def learn(
        self, batch: Batch, **kwargs: Any
    ) -> Dict[str, Union[float, List[float]]]:
        """Dispatch the data to all policies for learning.

        :return: a dict with the following contents:

        ::

            {
                "agent_1/item1": item 1 of agent_1's policy.learn output
                "agent_1/item2": item 2 of agent_1's policy.learn output
                "agent_2/xxx": xxx
                ...
                "agent_n/xxx": xxx
            }
        """
        results = {}
        for policy in self.policies:
            #data = batch[f"agent_{policy.agent_id}"]
            #if not data.is_empty():
            #data = self.get_agent_batch(batch, policy.agent_id-1)
            out = policy.learn(batch=batch["agent_" + str(policy.agent_id) ], batch_size = 16, repeat = 1, **kwargs)
            for k, v in out.items():
                results["agent_" + str(policy.agent_id) + "/" + k] = v
        return results
