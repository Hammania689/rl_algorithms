# HW4 PPO Implementation  by Hameed Abdul (hameeda2) :robot:
The goal of UIUC's AE 498 Reinforcment Learning Homework 4 is to get a working implementation of [PPO][ppo]

## Environment and Run Instructions :scroll:
- Scipy
- Matplotlib
- Pytorch >= 1.7.0
- Wandb
- Tqdm
- Gin-Config
- Numpy

## All Previous Runs Can Be Found On Wandb :sauna_man:
You can visualize all of my previous run [here][wandb]. When you run the code you will produce a new entry.

## Run Commands :running_man:
`python train_ppo.py --help`

![help]

Example [gin config][gin] usage. Basically one can pass optional and needed parameters to functions/classes and namespaces with a simple config file.

```python
train.num_iter = 3_000_000                                # Number of Training Iterations
train.num_trajs = 20                                      # Number of trajectories in batch 
train.num_actor_epochs = 20                               # Number of actor optimization steps each iteration
train.num_critic_epochs = 40                              # Number of critic optimization steps each iteration
train.nhid = 64                                           # Number of neurons in each hidden layer
train.bs = 20                                             # Batch size 
train.surrogate_objective = 'clip'			  # Surrogate loss function to use


Critic.lr = 1E-3                                          # Learning Rate for critic
```

---
# PPO with Dense Rewards :curling_stone:

```bash
python train_ppo.py config/dense.gin
```

### :bangbang: Preliminary results with ppo at 50k / 3 Million iterations :bangbang:
#### You can find an explanation of the results below [here](#dissecting-the-issues-microscope)

### Learning Curves
Actor (total reward) | Critic 
-|-
![Actor's loss][actor_lr] | ![Critic's loss][critic_lr]


### Policy and Value function
Actor's Learned Policy | Critic's Learned Value Function
-|-
![policy][policy] | ![value function][value]

### Example Trajectory
Plot | Video
-|-
![Trajectory plot][traj_plot] | ![Trajectory gif][traj_gif]

### Timing Stats :hourglass_flowing_sand:
![timing][timing]

---
# PPO with Sparse Rewards :rat:
```bash
python train_ppo.py config/sparse.gin
```

### :bangbang: Preliminary results with ppo at 40k / 3 Million iterations :bangbang:
#### You can find an explanation of the results below [here](#dissecting-the-issues-microscope)

### Learning Curves
Actor (total reward) | Critic 
-|-
![Actor's loss][actor_lr_sparse] | ![Critic's loss][critic_lr_sparse]


### Policy and Value function
Actor's Learned Policy | Critic's Learned Value Function
-|-
![policy][policy_sparse] | ![value function][value_sparse]

### Example Trajectory
Plot | Video
-|-
![Trajectory plot][traj_plot_sparse] | ![Trajectory gif][traj_gif_sparse]



# Dissecting the issues :microscope:
As you can see, my runs of ppo have yet to converge to stabilizing the pendelum in either reward case.

There are few reasons why this  might still be the case. 
1. There could still very well be a :bug: in my code. Given the preliminary success (trend in loss/total reward in the dense reward case), I have strayed away from this but could be possible.
2. The slow time per iteration (optimization + rollout batching logic) really adds up to slower convergence. Two possible solutions could be in refining the hyperparameters to achieve optimal agent faster and also refactoring my convoluded batching logic to be faster.

[Here][exp_log] is a spreadsheet that contians all the PPO training runs I have executed. Each entry contains a experiment description/question, expectation of results, notable configurations, actual result, notes to myself, and a link to the corresponding wandb run for visualizations. 

[ppo]: https://arxiv.org/pdf/1707.06347.pdf
[wandb]: https://wandb.ai/hammania689/ppo
[wandb_plug]: https://i.imgur.com/598Cal4.png
[help]: https://i.imgur.com/M03WYDp.png
[gin]: https://github.com/google/gin-config

[exp_log]: https://docs.google.com/spreadsheets/d/1BVPCyfK5TexD5LAe0LPDB3PJRFcznTaQ3beAMGie7xo/edit?usp=sharing

<!--Preliminary results go back and update-->
[traj_gif]: https://i.imgur.com/hmAqEbW.gif
[traj_plot]: https://i.imgur.com/VHYMuZt.png
[policy]: https://imgur.com/2oKYjwE.png
[value]: https://i.imgur.com/n16oTh0.png
[actor_lr]: https://i.imgur.com/Dg0pkYV.png
[critic_lr]: https://i.imgur.com/ffBDq8m.png 
[timing]: https://i.imgur.com/gAk2Nc1.png


[traj_gif_sparse]: https://i.imgur.com/mdnoUu6.gif
[traj_plot_sparse]: https://i.imgur.com/YtQTZoR.png
[policy_sparse]: https://i.imgur.com/cG2wtSg.png
[value_sparse]: https://i.imgur.com/ZLohauP.png
[actor_lr_sparse]: https://i.imgur.com/HHPoRB1.png
[critic_lr_sparse]: https://i.imgur.com/YMNSjfS.png
