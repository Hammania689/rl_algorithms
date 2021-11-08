import torch
from torch import nn

import ppo
from pendulum import Pendulum
from ppo.models import Actor, Critic


line_br = '=' * 50

def test_reshape():
    print(f"Testing the reshape application\n{line_br}")
    x = torch.randn((32, 32, 1))

    # Check that in this case we are only reshaping 
    assert (torch.equal(x.view(32, 32), x.sum(-1)) and 
                    torch.equal(x.squeeze(), x.sum(-1))), \
            ("Reshape Test Failed!")

    print("Status: Pass\n")




def test_rollout(env: Pendulum, 
                 actor: nn.Module,
                 critic: nn.Module,
                 num_steps: int = 100,
                 lambda_const: float = 0.95,
                 gamma: float = 0.99,
                 device: str = 'cpu'):
    
    # Comprehensive tensor that contains a batch of trajectories
    # Number of batchs x time step x (state cord 1, state cord 2, action, reward, value, advantages)
    eps = 0
    traj_batch = torch.zeros((1, num_steps, 6), requires_grad=False)
    

    advantages = torch.zeros((num_steps))
    v_targets = torch.zeros((num_steps))

    # Run policy with actor in environment to collect Trajectory
    traj_batch[eps] = ppo.utils.get_traj(env, actor, critic, num_steps, device)

    # Compute Advantages from current Trajectory
    # slice of rewards and value
    r_slice = torch.cat([traj_batch[eps,:,3], torch.zeros(1)])
    v_slice = torch.cat([traj_batch[eps,:,4], torch.zeros(1)])

    # δt = rt + γV (st+1 ) − V (st )
    delta = r_slice[:-1] + gamma * v_slice[1:] - v_slice[:-1]

    # At=δt+ (γλ)δt+1+···+···+ (γλ)T−t+1δT−1,
    advantages = delta.clone()

    for i in reversed(range(num_steps -1)):
        advantages[i] = advantages[i] + gamma * lambda_const * advantages[i+1]
    
    # compute value targets
    print(f"Testing Value Function Computation\n{line_br}")
    V = v_slice.clone()
    for t in reversed(range(num_steps)):
        # print(f'T: {t} | r_t: {r_slice[t]:.4f} | V_t+1: {V[t+1]:.4f}')
        V[t] = r_slice[t] + (gamma*V[t+1])
    V = V[:-1]

    v_targets = v_slice.clone()
    v_targets = torch.tensor([r_slice[t] + (gamma * v_slice[t+1]) for t in reversed(range(num_steps))])
            
    assert torch.allclose(V, v_targets), f"Value Function Computation failed!"
    
    v_cum_sum = torch.cumsum(v_slice, -1).flip(-1)
    v_targets = (r_slice[:-1].flip(-1) + gamma * v_cum_sum[1:]).flip(-1)
    assert torch.allclose(V, v_targets), f"Value Function Computation failed!"
    # update calculations into traj_batch
    traj_batch[eps,:,-1] = advantages
    traj_batch[eps,:, 4] = v_targets

    traj_dataset = TrajectoryDataset(traj_batch)
    total_reward = traj_batch[:, :, 3].mean().item()

    print("Status: Pass\n")


if __name__ == '__main__':
    actor = Actor()
    critic = Critic()

    env = Pendulum()
    test_reshape()
    test_rollout(env,actor, critic)
    
