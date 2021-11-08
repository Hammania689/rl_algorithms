import io 
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import torch
import wandb
from PIL import Image
from torch import nn

class Q(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 lr: float = 1E-3,
                 nhid: int = 64,
                 gamma: float = 0.95,
                 criterion: nn.Module = nn.MSELoss,
                 optim: torch.optim = torch.optim.Adam,
                 device: torch.device = 'cpu',
                ):

        super(Q, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, nhid),
            nn.Tanh(),
            nn.Linear(nhid, nhid),
            nn.Tanh(),
            nn.Linear(nhid, out_channels),
        )
        
        self.create_otpimizers(optim, lr)
        self.reset_meters()
        self.criterion = criterion()
        self.gamma = gamma

    def create_otpimizers(self, optim, lr):
        self.optim = optim(self.parameters(), lr)


    def reset_meters(self):
        self.loss_last = 0.0
        self.discount_reward_last = 0.0

        self.running_loss = []
        self.running_discounted_reward = []

    
    def plot_train_wandb(self, horizon_counter):
        wandb.log({
                'Noisy Loss': self.loss_last,
                'Noisy Total Discounted Reward': self.discount_reward_last,
                'Time': horizon_counter
               })


    def plot_eval_wandb(self, env, horizon_counter, device, filename='results_discreteaction_pendulum'):
        # Simulate an episode and save the result as an animated gif
        def policy(s):
            with torch.no_grad():
                s = torch.from_numpy(s).to(device).float()
                a = torch.argmax(self.forward(s))
                return a.cpu().numpy()
        env.video(policy, filename=filename+'.gif')

        # Initialize simulation
        s = env.reset()

        # Create dict to store data from simulation
        data = {
            't': [0],
            's': [s],
            'a': [],
            'r': [],
            'v': [],
        }

        # Simulate until episode is done
        done = False
        while not done:
            with torch.no_grad():
                q_block = self.forward(torch.from_numpy(env.s).float().to(device))
                a = torch.argmax(q_block, dim=-1).cpu()
                v = torch.max(q_block, dim=-1)[0].cpu()

            (s, r, done) = env.step(a)
            data['t'].append(data['t'][-1] + 1)
            data['s'].append(s)
            data['a'].append(a)
            data['r'].append(r)
            data['v'].append(v)

        # Parse data from simulation
        data['s'] = np.array(data['s'])
        theta = data['s'][:, 0]
        thetadot = data['s'][:, 1]
        tau = [env._a_to_u(a) for a in data['a']]

        # Plot data and save to png file
        fig, ax = plt.subplots(3, 1, figsize=(10, 10))
        ax[0].plot(data['t'], theta, label='theta')
        ax[0].plot(data['t'], thetadot, label='thetadot')
        ax[0].legend()
        ax[1].plot(data['t'][:-1], tau, label='tau')
        ax[1].legend()
        ax[2].plot(data['t'][:-1], data['r'], label='r')
        ax[2].legend()
        ax[2].set_xlabel('time step')
        plt.tight_layout()
        plt.savefig(filename + '.png')

        # Save gif locally and to wandb
        wandb.save(filename+'.gif')
        wandb.save(filename + '.png')
        val_fig = go.Figure()
        policy_fig = go.Figure()

        val_fig.add_trace(go.Scatter(x=theta.flatten(),
                             y=thetadot.flatten(),
                             marker={'color': np.array(data['v']).flatten(), 'showscale': True},
                             mode='markers'))

        policy_fig.add_trace(go.Scatter(x=theta.flatten(),
                             y=thetadot.flatten(),
                             marker={'color': np.array(data['a']).flatten(), 'showscale': True},
                             mode='markers'))

        def plotly_fig2array(fig):
            #convert Plotly fig to  an array
            fig_bytes = fig.to_image(format="png")
            buf = io.BytesIO(fig_bytes)
            img = Image.open(buf)
            return np.asarray(img)


        val_img = plotly_fig2array(val_fig)
        policy_img = plotly_fig2array(val_fig)
        wandb.log({'Learned Value Function': wandb.Image(val_img, caption=f'Agent at epoch {horizon_counter}'),
                   'Learned Policy': wandb.Image(policy_img, caption=f'Agent at epoch {horizon_counter}'),
                   'Avg Training Loss': sum(self.running_loss)/len(self.running_loss),
                   'Avg Total discounted reward': sum(self.running_discounted_reward)/len(self.running_discounted_reward),
                   'Example Gif': wandb.Video(filename+'.gif', caption=f'Agent at epoch {horizon_counter}', fps=10, format='gif'),
                   'Example Trajectory': wandb.Image(filename+'.png'),
                   'Time': horizon_counter})

        # Close plot so that we can avoid exstensive memory allocation
        plt.close()
        del val_fig


    def forward(self, x):
        return self.mlp(x)
    

    def optimize_params(self, data: torch.Tensor, target_q: nn.Module, device: str):
        """
            Optimization Step and Weight update for the model
        """
        self.train()
        
        # Send data to the device
        state_cord = data[:, :2]
        r = data[:, 3]
        state_prime_cord = data[:, -2:]
        
        state_cord.to(device)
        r.to(device)
        state_prime_cord.to(device)

        # Feed throught Network
        q_block = self.forward(state_cord)

        # Calculate y_j
        y_j = r + self.gamma * torch.max(target_q(state_prime_cord), -1)[0]

        # Clear out optimizer
        self.optim.zero_grad()

        # Calculate loss
        loss = self.criterion(q_block.max(dim=-1)[0], y_j)
        
        # Backprop and Update Weights
        loss.backward()

        # Adjust optimizer
        self.optim.step()

        # Add Noisy and Running stats to the respective meters
        discounted_reward = sum([self.gamma ** idx * reward  for idx, reward in enumerate(r)])

        self.loss_last = loss.item()
        self.discount_reward_last = discounted_reward

        self.running_loss.append(loss.item())
        self.running_discounted_reward.append(discounted_reward)
            
        

if __name__ == '__main__':
    q =  Q(88, 9)

    print(q)
