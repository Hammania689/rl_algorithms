import os
import time
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch import nn
from torch import distributions
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from pendulum import Pendulum

def get_rollouts(env: Pendulum, 
                        actor: nn.Module,
                        critic: nn.Module,
                        num_trajs: int,
                        num_steps: int,
                        bs: int,
                        lambda_const: float,
                        gamma: float,
                        device: str,
                        bootstrap_with_critic: bool = True,
                        hardcode_actor: bool = False,
                        normalize_advantages: bool = True):
    rollout_states, rollout_actions = [], []
    rollout_rewards = torch.tensor((), requires_grad=False).to(device)
    rollout_values = torch.tensor((), requires_grad=False).to(device)
    rollout_advantages = torch.tensor((), requires_grad=False).to(device)
    
    for eps in range(num_trajs):
        # Run policy with actor in environment to collect Trajectory
        traj_s, traj_a, traj_r, traj_v = get_traj(env, actor, critic, num_steps, device, hardcode_actor)

        # Slice of rewards and value estimates
        # Bootstrap with zero /critic's predicition of the terminating state's value  as specified
        if bootstrap_with_critic:
            traj_v.append(critic(torch.from_numpy(env.s).type(rollout_rewards.dtype).to(device)))
        else:
            traj_v.append(0)

        r = torch.tensor(traj_r).to(device)
        v = torch.tensor(traj_v).to(device)

        # Compute Advantages from current Trajectory
        # δt = rt + γV (st+1 ) − V (st )
        delta = r + gamma * v[1:] - v[:-1]

        # At=δt+ (γλ)δt+1+···+···+ (γλ)T−t+1δT−1,
        advantages = delta.clone()
        for i in reversed(range(num_steps -1)):
            advantages[i] = advantages[i] + gamma * lambda_const * advantages[i+1]
        
        # Compute value targets
        for t in reversed(range(num_steps)):
            v[t] = r[t] + (gamma*v[t+1])
        v = v[:-1]
        
        # Add updated traj into rollouts
        rollout_states.extend(traj_s)
        rollout_actions.extend(traj_a)

        rollout_rewards = torch.cat([rollout_rewards, r])
        rollout_values = torch.cat([rollout_values, v])
        rollout_advantages = torch.cat([rollout_advantages, advantages])

    # Calculate total reward
    total_reward = rollout_rewards.mean().item()

    rollout_states = torch.tensor(rollout_states, requires_grad=False).to(device)
    rollout_actions = torch.tensor(rollout_actions, requires_grad=False).to(device)
    
    # Update \pi_theta_old 
    actor.set_distr_old(rollout_states)

    # Normalize advantages
    if normalize_advantages:
        adv_mu, adv_std = torch.std_mean(advantages)
        rollout_advantages = (rollout_advantages - adv_mu) / adv_std

    rollout = (rollout_states, rollout_actions, rollout_values, rollout_advantages)
    return rollout, total_reward


def get_traj(env: Pendulum, 
             actor: nn.Module,
             critic: nn.Module,
             num_steps: int,
             device: str,
             hardcode_actor: bool = False):

    env.reset()
    
    # Hard coded policy 
    kP = 10.0
    kD = 1.0
    pd_policy = lambda s: torch.tensor([- kP * s[0] - kD * s[1]])
    
    s_t, a_t, r_t, v_t = [], [], [], []

    ref_tensor = torch.tensor(())
    with torch.no_grad():
        # Run policy with actor in environment to collect Trajectory
        for step in range(num_steps):
            s = torch.from_numpy(env.s).type(ref_tensor.dtype).to(device)
            mu, std = actor(s)
            if hardcode_actor:
                a = pd_policy(s)
            else:
                a = actor.distr_type(mu, std).sample().numpy()
            s_prime, r, _ = env.step(a)
            val = critic(s)

            s_t.append([s[0], s[1]])
            a_t.append(a)
            r_t.append(r)
            v_t.append(val)

    return s_t, a_t, r_t, v_t


def plot_figures(env, actor: nn.Module, critic: nn.Module, step: int, device: str,  dir_name: str = './results', hardcode_actor: bool = False) -> dict:
    """
    :Return: (dict) containing visualizations of both the learned value function and policy as well as a gif of an example trajectory
    """

    plot_log = {'step': step}
    
    # Plot example trajectory gif
    states, actions, rewards, values = get_traj(env, actor, critic, env.max_num_steps, device,  hardcode_actor)
    filename = os.path.join(dir_name, 'example_trajectory', f'pendulum_{step}.gif')
    with torch.no_grad():
        env.video(states, filename)
    plot_log.update({'Example Gif': wandb.Video(filename, caption=f'Agent at epoch {step}', fps=10, format='gif')})

    # Plot example trajectory plot
    states = np.array(states)
    theta = states[:, 0]
    thetadot = states[:, 1]
    tau = actions
    t = np.arange(len(states))
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    ax[0].plot(t, theta, label='theta')
    ax[0].plot(t, thetadot, label='thetadot')
    ax[0].legend()
    ax[1].plot(t, tau, label='tau')
    ax[1].legend()
    ax[2].plot(t, rewards, label='r')
    ax[2].legend()
    ax[2].set_xlabel('time step')
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name, 'trajectory', f'trajectory_{step}.png'))

    plot_log.update({'Example Trajectory': wandb.Image(plt, caption=f'Agent at epoch {step}')})
    plt.close()

    # Setup + Populate grid representing our "continous" state/value space
    n = 101
    theta = np.linspace(-np.pi, np.pi, n)
    thetadot = np.linspace(-env.max_thetadot, env.max_thetadot, n)
    u = np.empty((n, n))
    V = np.empty((n, n))

    for i in range(n):
        for j in range(n):
            s = np.array([theta[i], thetadot[j]])
            with torch.no_grad():
                s = torch.from_numpy(s).type(torch.tensor(()).dtype)
                V[j, i] = critic(s).item()
                u[j, i] = actor(s)[0].item()

    # Greedy Policy
    plt.figure()
    plt.pcolor(theta, thetadot, u, shading='nearest', vmin=-env.max_tau, vmax=env.max_tau)
    plt.xlabel('theta')
    plt.ylabel('thetadot')
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-env.max_thetadot, env.max_thetadot)
    cbar = plt.colorbar()
    cbar.set_label('tau')
    plt.title(f'step = {step:10d}')
    plt.savefig(os.path.join(dir_name, 'policy', f'policy_{step}.png'))
    
    plot_log.update(**{'Actor/Policy Visualization': wandb.Image(plt)})
    plt.close()

    # Learned Value Function 
    plt.figure()
    if env.sparse_reward:
        plt.pcolor(theta, thetadot, V, shading='nearest', vmin=0, vmax=100)    # <--- FIXME (?)
    else:
        plt.pcolor(theta, thetadot, V, shading='nearest', vmin=-100, vmax=0)
    plt.xlabel('theta')
    plt.ylabel('thetadot')
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-env.max_thetadot, env.max_thetadot)
    cbar = plt.colorbar()
    cbar.set_label('value function')
    plt.title(f'step = {step:10d}')
    plt.savefig(os.path.join(dir_name, 'value_function', f'value_function_{step}.png'))
    plot_log.update(**{'Critic/Learned Value Function Visualization': wandb.Image(plt)})
    plt.close()

    return plot_log

    
@contextmanager
def set_default_tensor_type(tensor_type):
        if torch.tensor(0).is_cuda:
            old_tensor_type = torch.cuda.FloatTensor
        elif torch.tensor == torch.FloatTensor:
            old_tensor_type = torch.FloatTensor
        else:
            old_tensor_type = torch.DoubleTensor
        
            torch.set_default_tensor_type(tensor_type)
            yield
            torch.set_default_tensor_type(old_tensor_type)

def log_times(optim_time: float, rollout_time: float,  log_time: float, step: int) -> dict:
    return {'Optimization time': optim_time,
            'Rollout Logic time': rollout_time,
            'Logging time': log_time,
            'step': step
           }
