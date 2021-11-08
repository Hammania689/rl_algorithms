import argparse
import time
from typing import Union

import gin
import numpy as np
import torch
import wandb
from tqdm import tqdm

import ppo
from pendulum import Pendulum
from ppo.models import Actor, Critic
from ppo.utils import get_rollouts, set_default_tensor_type


wandb.init(project="ppo")

# Train PPO
@gin.configurable
def train(env: Pendulum,
          num_iter: int,
          num_trajs: int, 
          num_actor_epochs: int,
          num_critic_epochs: int,
          surrogate_objective: str,
          nhid: int,
          bs: int,
          lambda_const: float = 0.95, 
          gamma: float = 0.95,
          save_interval: int = 10000,
          hardcode_actor: bool = False,
          normalize_advantages: bool = True,
          device: str = 'cpu'):

    actor = Actor(nhid, loss=surrogate_objective)
    critic = Critic(nhid)

    # Send Networks to device
    actor.to(device)
    critic.to(device)

    # Log gradients of each model
    wandb.watch(actor)
    wandb.watch(critic)

    if hardcode_actor:
        print(f"Debugging Critic with Hardcoded actor")

    for step in tqdm(range(num_iter)):

        # Construct dataloader of trajectories with current actor network 
        rollout_start = time.time()
        traj_batch, avg_total_reward = get_rollouts(env, actor, critic, num_trajs, env.max_num_steps, bs, lambda_const, gamma, device=device, hardcode_actor=hardcode_actor)
        rollout_total = time.time() - rollout_start
        
        # Reset logs for current iteration
        log = {}
        actor.reset_meters()
        critic.reset_meters()
        
        # Optimize actor network with surogate loss
        optim_start = time.time()
        if not hardcode_actor:
            for _ in range(num_actor_epochs):
                actor.optimize_params(traj_batch, device)

        # Optimize critic network with MSE loss
        for _ in range(num_critic_epochs):
            critic.optimize_params(traj_batch, device)

        optim_total = time.time() - optim_start

        # Periodically Save Actor/Critic checkpoints + example gifs
        log_start = time.time()
        if step % save_interval == 0:
            actor.save_model()
            critic.save_model()
            wandb.save('actor.ckpt')
            wandb.save('critic.ckpt')
            log.update(**ppo.utils.plot_figures(env, actor, critic, step, device=device, hardcode_actor=hardcode_actor))

        # Log and summary of iteration
        log.update(**critic.log_step(step))
        log.update(**actor.log_step(avg_total_reward, step))
        log_total = time.time() - log_start
        log.update(**ppo.utils.log_times(optim_total, rollout_total, log_total, step))

        wandb.log(log)
        tqdm.write(f"\n ({step + 1} / {num_iter}) Total reward: {avg_total_reward:.4f}" 
                  + f" | Critic loss: {np.mean(critic.avg_loss):.4f}"
                  + f" | Actor loss: {np.mean(actor.avg_loss):.4f}"
                  + f" | Rollout time: {rollout_total:.4f}s"
                  + f" | Optim time: {optim_total:.4f}s"
                  + f" | Log time: {log_total:.4f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reimplementation of PPO on Pendulum environment by Hameed Abdul', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='Gin config with hyperparameter settings')
    parser.add_argument('--cuda', action='store_true', help='Use gpu during training')
    parser.add_argument('--half_prec', action='store_true', help='Enable training with only half precision')

    args = parser.parse_args()

    # Parse train configurations and hyperparameters
    gin.parse_config_file(args.config)

    env = Pendulum()

    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        tensor_type = torch.cuda.DoubleTensor
        if args.half_prec:
            tensor_type = torch.cuda.FloatTensor
    else:
        device = torch.device('cpu')
        tensor_type = torch.DoubleTensor
        if args.half_prec:
            tensor_type = torch.FloatTensor

    with set_default_tensor_type(tensor_type):
        train(env=env)
