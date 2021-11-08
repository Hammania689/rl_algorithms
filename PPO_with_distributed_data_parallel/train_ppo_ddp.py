import argparse
import os
import json
import time
from typing import Union

import gin
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

import ppo
from pendulum import Pendulum
from ppo.models import Actor, Critic
from ppo.utils import get_rollouts, set_default_tensor_type

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# Train PPO
@gin.configurable
def train(rank: int,
          world_size: int,
          env: Pendulum,
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
          device: str = 'cpu',
          tensor_type: torch.TensorType = torch.FloatTensor,
          max_KL_per_iteration: float = 0.01):

    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    with set_default_tensor_type(tensor_type):
        actor = Actor(nhid, loss=surrogate_objective)
        critic = Critic(nhid)

        start_step = 0
       # try:
       #     # start_step = 50000
       #     # actor_ckpt = torch.load('./actor_50kbase.ckpt', map_location=device)
       #     # critic_ckpt = torch.load('./critic_50kbase.ckpt', map_location=device)
       #     
       #     start_step = 54200
       #     actor_ckpt = torch.load(f'./actor_{start_step}.ckpt', map_location=device)
       #     critic_ckpt = torch.load(f'./critic_{start_step}.ckpt', map_location=device)
       #     actor.load_state_dict(actor_ckpt)
       #     print(actor)
       #     critic.load_state_dict(critic_ckpt)
       #     print("Loaded sucessfully")
       # except Exception as e:
       #     start_step = 0
       #     print("Issue with loading checkpoints")
       #     print(e)
       #     exit()

        actor = actor.to(rank)
        critic = critic.to(rank)
        actor = DDP(actor, device_ids=[rank])
        critic = DDP(critic, device_ids=[rank])
        
        actor.module.reset_meters()
        # Send Networks to device

        # Log gradients of each model
        #wandb.watch(actor)
        #wandb.watch(critic)

        if hardcode_actor:
            print(f"Debugging Critic with Hardcoded actor")

        log_history = []
        critic_losses = []
        rewards = []

        for step in tqdm(range(start_step, num_iter)):

            # Construct dataloader of trajectories with current actor network 
            rollout_start = time.time()
            traj_batch, avg_total_reward = get_rollouts(env, actor, critic, num_trajs, env.max_num_steps, bs, lambda_const, gamma, device=rank, hardcode_actor=hardcode_actor)

            avg_total_reward = [torch.tensor(avg_total_reward).to(rank)]
            combined_states = [torch.zeros((traj_batch[0].shape)).to(rank) for _ in range(world_size)]
            combined_actions = [torch.zeros((traj_batch[1].shape)).to(rank) for _ in range(world_size)]
            combined_values = [torch.zeros((traj_batch[2].shape)).to(rank) for _ in range(world_size)]
            combined_advantages = [torch.zeros((traj_batch[3].shape)).to(rank) for _ in range(world_size)]
            combined_rewards = [torch.zeros((1)).to(rank) for _ in range(world_size)]


            dist.all_gather_multigpu(combined_states, traj_batch[0])
            dist.all_gather_multigpu(combined_actions, traj_batch[1])
            dist.all_gather_multigpu(combined_values, traj_batch[2])
            dist.all_gather_multigpu(combined_advantages, traj_batch[3])

            # This operation only takes the sum
            # Must divide by world_size to get the mean
            dist.all_reduce_multigpu(avg_total_reward)
            avg_total_reward = avg_total_reward.pop().item() / world_size

            combined_batch = (combined_states[rank], combined_actions[rank], combined_values[rank], combined_advantages[rank])

            rollout_total = time.time() - rollout_start
            
            # Reset logs for current iteration
            log = {}
            actor.module.reset_meters()
            critic.module.reset_meters()
            
            # Optimize actor network with surogate loss
            optim_start = time.time()
            if not hardcode_actor:
                for _ in range(num_actor_epochs):
                    kl_ratio = actor.module.optimize_params(combined_batch, rank)

                    if kl_ratio > 1.5 * max_KL_per_iteration:
                        break

            # Optimize critic network with MSE loss
            for _ in range(num_critic_epochs):
                critic.module.optimize_params(combined_batch, rank)

            optim_total = time.time() - optim_start

            # Periodically Save Actor/Critic checkpoints + example gifs
            if rank == 0:
                log_start = time.time()

                if step % save_interval == 0:
                    actor.module.save_model(f'actor_{step}.ckpt')
                    critic.module.save_model(f'critic_{step}.ckpt')
                    #wandb.save('actor.ckpt')
                    #wandb.save('critic.ckpt')
                    log.update(**ppo.utils.plot_figures(env, actor, critic, step, device=rank, hardcode_actor=hardcode_actor))
                    with open('log_history_ddp.json','w') as j:
                        json.dump(log_history, j)
                    if step != 0:
                        ppo.utils.final_plots(critic_losses, rewards, list(range(step)), step)
                log.update(**critic.module.log_step(step))
                log.update(**actor.module.log_step(avg_total_reward, step))

                critic_losses.append(np.mean(critic.module.avg_loss))
                rewards.append(avg_total_reward)

                log_total = time.time() - log_start
                log.update(**ppo.utils.log_times(optim_total, rollout_total, log_total, step))
                tqdm.write(f"\n ({step + 1} / {num_iter}) Total reward: {avg_total_reward:.4f}" 
                  + f" | Critic loss: {np.mean(critic.module.avg_loss):.4f}"
                  + f" | Actor loss: {np.mean(actor.module.avg_loss):.4f}"
                  + f" | Rollout time: {rollout_total:.4f}s"
                  + f" | Optim time: {optim_total:.4f}s"
                  + f" | Log time: {log_total:.4f}s")


            log_history.append(log)

    if rank == 0:
        with open('log_history_ddp.json','w') as j:
            # log_history = json.dumps(log_history)
            json.dump(log_history, j)

        ppo.utils.final_plots(critic_losses, rewards, list(range(num_iter)), step)

    cleanup()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reimplementation of PPO on Pendulum environment by Hameed Abdul', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='Gin config with hyperparameter settings')
    parser.add_argument('world_size', type=int, default=2, help='Number of gpus per node')
    parser.add_argument('--seed', type=int, default=42 , help='Seed for random number generator') 
    parser.add_argument('--cuda', action='store_true', help='Use gpu during training')
    parser.add_argument('--half_prec', action='store_true', help='Enable training with only half precision')

    args = parser.parse_args()

    # Parse train configurations and hyperparameters
    gin.parse_config_file(args.config)
    torch.manual_seed(args.seed)

    env = Pendulum()
    world_size = args.world_size

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

    if torch.cuda.device_count() > 1:
       print("We have available ", torch.cuda.device_count(), "GPUs! but using ",world_size," GPUs")

    mp.spawn(train, args=(world_size, env, int(1E6), 10, 20, 40, 'clip', 100, 32, 0.95, 0.95, 100, False, True,device, tensor_type), nprocs=world_size, join=True)

