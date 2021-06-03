#!/usr/bin/env python
# Created at 2020/2/9

import click
import torch
from torch.utils.tensorboard import SummaryWriter

from Algorithms.pytorch.DQN.dqn import DQN


@click.command()
@click.option("--env_id", type=str, default="MountainCar-v0", help="Environment Id")
@click.option("--render", type=bool, default=False, help="Render environment or not")
@click.option("--num_process", type=int, default=1, help="Number of process to run environment")
@click.option("--lr", type=float, default=1e-3, help="Learning rate for Policy Net")
@click.option("--gamma", type=float, default=0.99, help="Discount factor")
@click.option("--epsilon", type=float, default=0.90, help="Probability controls greedy action")
@click.option("--explore_size", type=int, default=5000, help="Explore steps before execute deterministic policy")
@click.option("--memory_size", type=int, default=100000, help="Size of replay memory")
@click.option("--step_per_iter", type=int, default=1000, help="Number of steps of interaction in each iteration")
@click.option("--batch_size", type=int, default=128, help="Batch size")
@click.option("--min_update_step", type=int, default=1000, help="Minimum interacts for updating")
@click.option("--update_target_gap", type=int, default=50, help="Steps between updating target q net")
@click.option("--max_iter", type=int, default=500, help="Maximum iterations to run")
@click.option("--eval_iter", type=int, default=50, help="Iterations to evaluate the model")
@click.option("--save_iter", type=int, default=50, help="Iterations to save the model")
@click.option("--model_path", type=str, default="trained_models", help="Directory to store model")
@click.option("--log_path", type=str, default="../log/", help="Directory to save logs")
@click.option("--seed", type=int, default=1, help="Seed for reproducing")

# env_id: 模拟环境
# num_process: 运行环境的进程
# explore_size: 探索长度
# memory_size: 数据存储池（replay buffer）
# step_per_iter: 每一次迭代的interactions
# max_iter: 迭代次数
def main(env_id, render, num_process, lr, gamma, epsilon, explore_size, memory_size, step_per_iter, batch_size,
         min_update_step, update_target_gap, max_iter, eval_iter, save_iter, model_path, log_path, seed):
    base_dir = log_path + env_id + "/DQN_exp{}".format(seed)
    # 记录日志
    writer = SummaryWriter(base_dir)

    # 定义DQN
    dqn = DQN(env_id,
              render=render,
              num_process=num_process,
              memory_size=memory_size,
              lr_q=lr,
              gamma=gamma,
              epsilon=epsilon,
              explore_size=explore_size,
              step_per_iter=step_per_iter,
              batch_size=batch_size,
              min_update_step=min_update_step,
              update_target_gap=update_target_gap,
              seed=seed)

    for i_iter in range(1, max_iter + 1):
        # 学习
        dqn.learn(writer, i_iter)

        if i_iter % eval_iter == 0:
            # 评估
            dqn.eval(i_iter, render=render)

        if i_iter % save_iter == 0:
            # 保存
            dqn.save(model_path)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
