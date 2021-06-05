import time
import tqdm
import numpy as np
from collections import defaultdict
from typing import Dict, Union, Callable, Optional

from tianshou.data import Collector
from tianshou.policy import BasePolicy
from tianshou.utils import tqdm_config, MovAvg, BaseLogger, LazyLogger
from tianshou.trainer import test_episode, gather_info
from tool import Tracker, Utils
from utils import HIGHLIGHT, SUCCESS


def offpolicy_trainer(
    policy: BasePolicy,
    train_collector: Collector,
    test_collector: Collector,
    utils: Utils,
    max_epoch: int,
    step_per_epoch: int,
    step_per_collect: int,
    episode_per_test: int,
    batch_size: int,
    update_per_step: Union[int, float] = 1,
    train_fn: Optional[Callable[[int, int], None]] = None,
    test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
    stop_fn: Optional[Callable[[float], bool]] = None,
    save_fn: Optional[Callable[[BasePolicy], None]] = None,
    reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    test_in_train: bool = True
) -> Dict[str, Union[float, str]]:
    """A wrapper for off-policy trainer procedure.

    The "step" in trainer means an environment step (a.k.a. transition).

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
    :param Collector train_collector: the collector used for training.
    :param Collector test_collector: the collector used for testing.
    :param int max_epoch: the maximum number of epochs for training. The training
        process might be finished before reaching ``max_epoch`` if ``stop_fn`` is set.
    :param int step_per_epoch: the number of transitions collected per epoch.
    :param int step_per_collect: the number of transitions the collector would collect
        before the network update, i.e., trainer will collect "step_per_collect"
        transitions and do some policy network update repeatly in each epoch.
    :param episode_per_test: the number of episodes for one policy evaluation.
    :param int batch_size: the batch size of sample data, which is going to feed in the
        policy network.
    :param int/float update_per_step: the number of times the policy network would be
        updated per transition after (step_per_collect) transitions are collected,
        e.g., if update_per_step set to 0.3, and step_per_collect is 256, policy will
        be updated round(256 * 0.3 = 76.8) = 77 times after 256 transitions are
        collected by the collector. Default to 1.
    :param function train_fn: a hook called at the beginning of training in each epoch.
        It can be used to perform custom additional operations, with the signature ``f(
        num_epoch: int, step_idx: int) -> None``.
    :param function test_fn: a hook called at the beginning of testing in each epoch.
        It can be used to perform custom additional operations, with the signature ``f(
        num_epoch: int, step_idx: int) -> None``.
    :param function save_fn: a hook called when the undiscounted average mean reward in
        evaluation phase gets better, with the signature ``f(policy:BasePolicy) ->
        None``.
    :param function stop_fn: a function with signature ``f(mean_rewards: float) ->
        bool``, receives the average undiscounted returns of the testing result,
        returns a boolean which indicates whether reaching the goal.
    :param function reward_metric: a function with signature ``f(rewards: np.ndarray
        with shape (num_episode, agent_num)) -> np.ndarray with shape (num_episode,)``,
        used in multi-agent RL. We need to return a single scalar for each episode's
        result to monitor training in the multi-agent RL setting. This function
        specifies what is the desired metric, e.g., the reward of agent 1 or the
        average reward over all agents.
    :param BaseLogger logger: A logger that logs statistics during
        training/testing/updating. Default to a logger that doesn't log anything.
    :param bool verbose: whether to print the information. Default to True.
    :param bool test_in_train: whether to test in the training phase. Default to True.

    :return: See :func:`~tianshou.trainer.gather_info`.
    """
    env_step, gradient_step = 0, 0
    start_time = time.time()
    train_collector.reset_stat()
    test_collector.reset_stat()
    test_in_train = test_in_train and train_collector.policy == policy
    best_reward, best_reward_std = None, None
    rewards = None
    len = 0
    tracker = utils.get_tracker()

    for epoch in range(1, 1 + max_epoch):
        # train
        policy.train()
        with tqdm.tqdm(total=step_per_epoch) as t:
            while t.n < t.total:
                if train_fn:
                    train_fn(epoch, env_step)
                # collect data
                result = train_collector.collect(n_step=step_per_collect)
                # track episode data
                for ep in range(result['n/ep']):
                    tracker.track('train_episode',{
                        'len': int(result['lens'][ep]),
                        'rew': result['rews'][ep].tolist()
                    })
                    len = result['lens'][ep]
                    rewards = result['rews'][ep]

                env_step += int(result["n/st"])
                t.update(result["n/st"])

                # early stopping
                if result["n/ep"] > 0:
                    if test_in_train and stop_fn and stop_fn(result["rew"]):
                        test_result = test_episode(
                            policy, test_collector, test_fn,
                            epoch, episode_per_test, None, env_step)
                        if stop_fn(test_result["rew"]):
                            if save_fn:
                                save_fn(policy)
                            return gather_info(
                                start_time, train_collector, test_collector,
                                test_result["rew"], test_result["rew_std"])
                        else:
                            policy.train()
                for i in range(round(update_per_step * result["n/st"])):
                    gradient_step += 1
                    losses = policy.update(batch_size, train_collector.buffer) #batch_size=batch_size,repeat = 10)
                    tracker.track('train_losses', list(losses.values()))
                describe = HIGHLIGHT(f"#{epoch} ")
                if rewards is not None:
                    for index, rew in enumerate(rewards):
                        describe+= SUCCESS(f'r{index+1}: ') + "{:.2f} ".format(rew)
                    for index, loss in enumerate(losses.values()):
                        describe+= HIGHLIGHT(f"l{index+1}: ") + "{:.2f} ".format(loss)
                    describe+= "len: {}".format(len)
                t.set_description(desc=describe)
            if t.n <= t.total:
                t.update()
            # show
        # test
        #from tianshou.trainer import offpolicy_trainer
        test_result = test_episode(policy, test_collector, test_fn, epoch,
                                   episode_per_test, None, env_step, reward_metric)
        rew  = test_result["rews"]
        rew_mean = rew.mean(axis = 0)
        rew_std = rew.std(axis = 0)
        for ep in range(test_result['n/ep']):
            tracker.track('test_episode',{
                'len': int(test_result['lens'][ep]),
                'rew': test_result['rews'][ep].tolist()
            })

        # ??? to be update
        save_fn(policy)

        describe = HIGHLIGHT(f'Epoch #{epoch} Rewards:\n')
        for index, (rew_mean_i, rew_std_i) in enumerate(zip(rew_mean, rew_std)):
            describe += "ag{}: {:.4f} ± {:.4f}\n".format(index+1, rew_mean_i, rew_std_i)
        utils.log(describe)
        utils.log('++++++++++++++++++++++++++++')
        tracker.save()

        if stop_fn and stop_fn(best_reward):
            break
