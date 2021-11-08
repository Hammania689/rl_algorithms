# HW5 PPO Parrallization Implementation  by Hameed Abdul (hameeda2) :robot:
The goal of UIUC's AE 498 Reinforcment Learning Homework 5 is to get a working parrallized implementation of [PPO][ppo]

# Environment and Run Instructions :scroll:
- Scipy
- Matplotlib
- Pytorch >= 1.7.0
- Tqdm
- Gin-Config
- Numpy


# What is PyTorch Distributed Data Paralell (DDP)? And Why did I make use of it?

## Full Parallization Across Multi-GPUs
<!-- PyTorch has many ways to leverage computational resources across different hardware (GPU/CPU). In terms of parallelism, PyTorch's [Data Parallel][dp] is a straightforward to leverage more than a single GPUs during training. However, Data Parallel is a single process and multithreaded. One can make use of [Pytorch's Multiprocessing][multiproc] for a parallelized interface with more control. But with that power comes much responsiblility and ways to shoot oneself in the foot.  -->

PyTorch's [Distributed Data Parallel (DDP)][ddp] is designed to parallize computational resources in a distributed manner. This can be done either on several machines or a single machine with the options of single gpu, multi-gpu or strictly cpu parallization. (It should be noted that other methods of gpu acceleration/parallelism within PyTorch such as [PyTorch's Data parallel][dp] only uses multi-threading or [PyTorch's multiprocessing][multiproc] alone has many caveats to be aware of)

When launch, during the training process DDP automatically syncs the model's gradients and buffers accross all processes. DDP's api also provides a straight forward way to broadcast and communication between processes.  
### Why Not Just Use MPI?
My rational for making use of DDP over MPI, is that MPI requires more setup to take advantage of Cuda in Pytorch ([you must build Pytorch from source and enable mpi cuda functionality][mpi_setup]) and logic on the user's part (having to coordinate when to broadcast/reduce as well as synchronize gradients for each instance of the model being trained) and MPI is consistently outperformed by [NCCL][nccl] when it comes to parallel GPU usage.
Furthermore for my implementation of PPO the optimization step was the most time consuming aspect of training which could surely be sped up with GPU usage,
[which in retrospect raises concern with some of my original implementation choices][hw4]

### Pros and Cons of DDP 
  - Pros
    - Automatic gradient reduction between all models
    - Nice wrapping of several parallelization libraries allow more control (e.g MPI, NCCL, Gloo)
  - Cons
    - You can still shoot yourself in the foot if the API is used incorrectly
    - [Instability][ddp_unstable] that can occur during training makes its use questionable at times
    - Making use of [Logging][ddp_log] tools like Tensorboard and Wandb can be a bit tricky



[dp]: https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
[multiproc]: https://pytorch.org/docs/stable/multiprocessing.html#module-torch.multiprocessing
[ddp]: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

[ddp_log]:https://github.com/wandb/client/issues/452
[ddp_unstable]: https://www.reddit.com/r/MachineLearning/comments/egswuk/d_need_advise_with_pytorch_distributed_setup/
[nccl]: https://developer.nvidia.com/nccl
[mpi_setup]: https://pytorch.org/tutorials/intermediate/dist_tuto.html

## Timing Statistics Comparison

A comparison of different execution configurations using the same hyperparameters/experiment config.

| | Average Iteration time (secs)|
|-|-|
|Serial CPU| ![cpu]
| Serial 1 GPU| ![gpu]
| **DDP 2 GPUs** (Averaged stats) | Rollout Batching: 1.2910s , Optimization Steps: 0.0989s, Logging: 0.0001s 

[cpu]: https://i.imgur.com/tbzio3B.png
[gpu]: https://i.imgur.com/GZeamhu.png


# Results on Pendulum Environment

### :bangbang: Refer to [my HW4 submission][hw4] for more detail on performance discrepencies :bangbang:

[hw4]:https://github.com/compdyn/598rl-fa20/tree/hw4_hameeda2/hw4/hw4_hameeda2#dissecting-the-issues-microscope

## Run Commands :running_man:
`python train_ppo_ddp.py --help`

![help]

Example [gin config][gin] usage. Basically one can pass optional and needed parameters to functions/classes and namespaces with a simple config file.

```python
train.num_iter = 3_000_000                                # Number of Training Iterations
train.num_trajs = 20 / number of processes                # Number of trajectories in batch 
train.num_actor_epochs = 20                               # Number of actor optimization steps each iteration
train.num_critic_epochs = 40                              # Number of critic optimization steps each iteration
train.nhid = 64                                           # Number of neurons in each hidden layer
train.bs = 20                                             # Batch size 
train.surrogate_objective = 'clip'			              # Surrogate loss function to use


Critic.lr = 1E-3                                          # Learning Rate for critic
```

__* `number of processes` or `world_size` are set by the number of gpus included in each DDP run__

---

# DDP PPO with Dense Rewards :curling_stone:

```bash
python train_ppo_ddp.py config/dense.gin {num of gpus} 
```

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

---
# PPO with Sparse Rewards :rat:
```bash
python train_ppo_ddp.py config/sparse.gin {num of gpus}
```

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

# Conclusion
As seen, in both dense/sparse reward cases my DDP training is very finicky and would require much more care when training.
However, the bulk of PPO's poor performance could be attribute to unresolved issues in [my serial implementation][hw4]. 

[ppo]: https://arxiv.org/pdf/1707.06347.pdf
[wandb]: https://wandb.ai/hammania689/ppo
[wandb_plug]: https://i.imgur.com/598Cal4.png
[help]: https://i.imgur.com/mpvDKva.png
[gin]: https://github.com/google/gin-config

[traj_gif]: https://i.imgur.com/YGuBN6U.gif
[traj_plot]: https://i.imgur.com/l4Hma5q.png
[policy]: https://i.imgur.com/xLED72O.giff
[value]: https://i.imgur.com/YJt6kyh.giff
[actor_lr]: https://i.imgur.com/Rn1joHF.png
[critic_lr]: https://i.imgur.com/jC2quM9.png



[traj_gif_sparse]: https://i.imgur.com/AUT4VU7.gif
[traj_plot_sparse]: https://i.imgur.com/DbvR4Y9.png
[policy_sparse]: https://i.imgur.com/CTuuPW7.png
[value_sparse]: https://i.imgur.com/N8NI6vd.pngg
[actor_lr_sparse]: https://i.imgur.com/xjUS5Aq.png
[critic_lr_sparse]: https://i.imgur.com/f0fvpy3.pngg
