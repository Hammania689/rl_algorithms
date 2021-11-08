import argparse
import math

import numpy as np
import torch
import wandb
from tqdm import tqdm

from discreteaction_pendulum import Pendulum
from algorithm import Q


parser = argparse.ArgumentParser(description='Reimplementation of DQN on Pendulum environment by Hameed Abdul', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--epochs', default=100000, help='Number of training epochs')
parser.add_argument('--bs', default=32, help='Batch size for training')
parser.add_argument('--reset_interval', default=2000, help='How often to reset the target network')
parser.add_argument('--replay_init', default=1000, help='Initial sample size of replay experiences')
parser.add_argument('--replay_size', default=10000, help='Limit of memory replay buffer')

parser.add_argument('--epsilon', default=0.2, help='Epsilon determines how often to do an random action')
parser.add_argument('--gamma', default=0.95, help='Gamma discount factor')
parser.add_argument('--lr', default=1e-3, help='Learning Rate')
parser.add_argument('--device', default='cuda', help='Device to train model on')
parser.add_argument('--tag', default='vanilla', help=('DQN ablation tag to help sort experiments. \n' +
                    'vanilla: regular dqn \n' +
                    'targetless: dqn without target\n' +
                    'replayless: dqn without replay\n'))

args = parser.parse_args()

# hyper params
batch_size = args.bs
reset_q = args.reset_interval 
replay_size = args.replay_size
replay_init_sample = args.replay_init if args.replay_init < replay_size else replay_size
eps = args.epsilon
episodes = args.epochs
lr = args.lr

# Experiment Log Management
wandb.init(project="dqn", tags=args.tag.split())

device = 'cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu'
if device == 'cpu':
    torch.set_default_tensor_type(torch.FloatTensor)
else:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


def populate_replay(env: Pendulum, pop_size: int):
    """
    Populate the replay memory buffer with random policy
    """
    env.reset()
    s = env.s
    a_uni = torch.distributions.Uniform(0, env.num_actions)

    def sample_and_use(env): 
        a = math.floor(a_uni.sample())
        s_1, s_2 = torch.from_numpy(env.s).float().to(device)
        s_prime_cords, r = env.step(a)[:-1]
        return (s_1, s_2, a, torch.tensor(r)) +  tuple(torch.from_numpy(s_prime_cords).float().to(device))
    return torch.tensor([sample_and_use(env) for _ in range(pop_size)])


def stack_samples(env: Pendulum, bs: int, eps: float, QNet: torch.nn.Module, replay_memory: torch.tensor, horizon_counter, device: str):
    """
    Sample a N number of replay samples form the environment, where N is the batch size
    """
    s = env.s
    stack = []
    for _ in range(bs):
        # Epsilon greedy select a random action
        if torch.rand(1) < eps:
            a = torch.randint(env.num_actions, (1,)).item()
        else:
            a = torch.argmax(QNet(torch.from_numpy(s).float().to(device)), -1).item()

        s_prime, r, _ = env.step(a)

        # Populate the replay memory
        replay_size = replay_memory.shape[0]
        stack.append([s[0], s[1], a , r, s_prime[0], s_prime[1]])
        s = s_prime
    
    modulo_bound = (((horizon_counter * bs) % replay_size) + replay_size) % replay_size
    replay_batch_slice = replay_memory[modulo_bound:modulo_bound + 32]
    stack = torch.tensor(stack).to(device).float()

    # Overwrite N entries
    if replay_batch_slice.shape[0] < bs:
        # In the case that we need to circle from end back to beginning
        end_slice = replay_batch_slice.shape[0]
        replay_memory[modulo_bound: modulo_bound + end_slice] = stack[:end_slice]

        start_slice = bs - end_slice
        replay_memory[:start_slice] = stack[end_slice:]
    else:
        replay_memory[modulo_bound:modulo_bound + 32] = stack


if __name__ == '__main__':

    # Initialize the environment and other variables
    env = Pendulum(rg=np.random.RandomState())
    reset_counter = 0
    horizon_counter = 0
    replay_memory = torch.zeros((replay_size, 6))
    replay_memory[-replay_init_sample:, :] = populate_replay(env, replay_init_sample)
    
    # Initialize Target and Q networks + Send Models to device
    QNet = Q(in_channels=2, out_channels=env.num_actions, lr=lr)
    Q_target = Q(in_channels=2, out_channels=env.num_actions)

    QNet.to(device)
    Q_target.to(device)
    
    # Log model gradients, parameters, etc
    wandb.watch(QNet)

    for _ in tqdm(range(episodes)):
        
        # Reset the environment at the start of each episode
        env.reset()

        for t in range(env.max_num_steps):
            # Random mini_batch Sample of replay buffer
            stack_samples(env, batch_size, eps, QNet, replay_memory, horizon_counter, device)

            # Only sample from the populated entries in the replay memory buffer
            populated_mask = ~torch.tensor([torch.equal(row_slice, torch.zeros((6))) for row_slice in replay_memory])
            masked_replay = replay_memory[populated_mask]
            rand_sample = torch.randint(len(masked_replay), size=(batch_size,))
            mini_batch = masked_replay[rand_sample]

            # Optimize the Q network's parameters
            QNet.optimize_params(data=mini_batch, target_q=Q_target, device=device)
            QNet.plot_train_wandb(horizon_counter)

            reset_counter += 1
            horizon_counter += 1

            if reset_counter == reset_q:
                # Set the current target network's weights to that of the Q Network
                Q_target.load_state_dict(QNet.state_dict())
                reset_counter = 0

        # Log averaged metrics and clear out meters
        QNet.plot_eval_wandb(env, horizon_counter, device)
        QNet.reset_meters()
