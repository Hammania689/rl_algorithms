from typing import Union

import gin
import numpy as np
import torch
import wandb
from torch import nn
from torch import distributions


def base_loss(distr: torch.distributions, distr_old: torch.distributions, a: torch.Tensor, advantages: torch.Tensor):
    """r
    L_t(\theta) = r_t(\theta)*A\hat_t
    where r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_\theta_old(a_t|s_t)}
    """
    return - torch.mean(torch.exp(distr.log_prob(a) - distr_old.log_prob(a)) * advantages)

def kld_loss():
    pass

def clipped_loss(distr: torch.distributions, distr_old: torch.distributions, a: torch.Tensor, advantages: torch.Tensor, eps: float = 0.2):
    r_t = torch.exp(distr.log_prob(a) - distr_old.log_prob(a))
    cpi = (r_t * advantages)
    clipped_cpi = (torch.clip(r_t, 1 - eps, 1 + eps) * advantages)
    return - torch.minimum(cpi, clipped_cpi).mean()


@gin.configurable
class Actor(nn.Module):
    def __init__(self,
                 nhid: int = 64,
                 lr: float = 1E-4,
                 loss: str = 'base',
                 episolon: float = 0.2,
                 distr_type: distributions = distributions.Normal):
        super(Actor, self).__init__()

        self.std = nn.Parameter(-0.5 * torch.ones(1))
        self.mlp =  nn.Sequential(nn.Linear(2, nhid),
                                  nn.Tanh(),
                                  nn.Linear(nhid, nhid),
                                  nn.Tanh(),
                                  nn.Linear(nhid, 1))

        self.distr_type = distr_type
        self.episolon = episolon
        self.create_optim(lr=lr)

        loss_config = {'base': base_loss,
                       'kld': kld_loss,
                       'clip': clipped_loss}

        self.surrogate_loss = loss_config[loss]

    def set_distr_old(self, states):
        with torch.no_grad():
            mu, std = self.forward(states)
        self.distr_old = self.distr_type(mu, std)

    def create_optim(self, lr: float):
        self.optim = torch.optim.Adam(params=self.parameters(),lr=lr)


    def forward(self, s: torch.Tensor):
        """
        :param s: (torch.Tensor) Input state
        :param device: (str) Device to send tensors
        :return: Mean and Std from policy
        """
        mu = torch.squeeze(self.mlp(s))
        std = torch.exp(self.std)
        return mu, std


    def optimize_params(self, batch: list, device: str):

        loss_meter = []
        
        state_coords, a,  _, advantages = batch
        state_coords = state_coords.to(device)
        a = a.to(device)
        advantages = advantages.to(device)

        mu, std = self.forward(state_coords)

        loss = self.surrogate_loss(self.distr_type(mu, std), self.distr_old, a, advantages, self.episolon)
        loss.backward()

        loss_meter.append(loss.item())
        self.optim.step()

        self.optim.zero_grad()

        self.avg_loss.extend(loss_meter)
        wandb.log({'Actor/Noisy/Loss': np.mean(loss_meter)})


    def reset_meters(self):
        self.avg_loss = []


    def log_step(self, avg_total_reward: float, step: int):
        if self.avg_loss == []:
            self.avg_loss = 1E-7
        return {'Actor/Averaged/Loss': np.mean(self.avg_loss),
                'Actor/Averaged/Total Reward': avg_total_reward,
                'step': step}


    def save_model(self, filename='actor.ckpt'):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))

@gin.configurable
class Critic(nn.Module):
    def __init__(self,
                 nhid: int = 64,
                 loss: nn.Module = nn.MSELoss(),
                 lr: float = 1E-4):
        super(Critic, self).__init__()

        self.mlp =  nn.Sequential(nn.Linear(2, nhid),
                                  nn.Tanh(),
                                  nn.Linear(nhid, nhid),
                                  nn.Tanh(),
                                  nn.Linear(nhid, 1))
        
        self.loss = loss
        
        # Create optimizer
        self.create_optim(lr)


    def create_optim(self, lr):
        self.optim = torch.optim.Adam(params=self.parameters(), lr=lr)


    def forward(self, s):
        """
        :param s: Input State 
        :return: Value score for the input state 
        """
        v = torch.squeeze(self.mlp(s))
        return v


    def optimize_params(self, batch: list, device: str):
        
        loss_meter = []

        state_coords, _, v_targets, _ = batch
        state_coords = state_coords.to(device)
        v_targets = v_targets.to(device)

        v = self.forward(state_coords)

        loss = self.loss(v, v_targets)
        loss.backward()

        self.optim.step()

        self.optim.zero_grad()

        loss_meter.append(loss.item())

        self.avg_loss.extend(loss_meter)
        wandb.log({'Critic/Noisy/Loss': np.mean(loss_meter)})


    def reset_meters(self):
        self.avg_loss = []


    def log_step(self, step: int):
        return {'Critic/Averaged/Loss': np.mean(self.avg_loss),
                'step': step}

    def save_model(self, filename='critic.ckpt'):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))

