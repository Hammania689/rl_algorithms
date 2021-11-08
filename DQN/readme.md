# HW2 AE598 Solution Hameed Abdul (hameeda2) :robot:
Hameed Abdul's implementation of [DQN][dqn]

# Environment and Run Instructions :scroll:
- Scipy
- Matplotlib
- Pytorch >= 1.3.0
- Wandb
- Tqdm
- Plotly

## All Previous Runs Can Be Found On Wandb :sauna_man:
You can visualize all of my previous run [here][wandb]. When you run the code you will produce a new entry.
![wandb_plug]

### Running single run using vanilla dqn 
`python train_ dqn.py`

![help]

### Running all albation experiments
`./ablation_exp.sh`

---
# Vanilla DQN Results 

## Learning Curves
![lr_curve]

## Example Trajectory
### Plot
![trajectory]

### Gif 
![policy_gif]


## Policy Visualization 
![policy]

## Value Function Visualization
![value]

---

# Ablation Study :space_invader:

## Based on Table 3 from the [paper][dqn]
![table3]

[table3]: https://i.imgur.com/I2k5knk.png
[dqn]: https://www.nature.com/articles/nature14236
[wandb]: https://wandb.ai/hammania689/dqn
[wandb_plug]: https://i.imgur.com/598Cal4.png
[help]: https://i.imgur.com/6RZxwsk.png
[policy_gif]: https://i.imgur.com/8rm92Oz.gif
[trajectory]: https://i.imgur.com/HbLQjW7.png
[lr_curve]: https://i.imgur.com/pFUWuyt.png
[value]: https://api.wandb.ai/files/hammania689/dqn/2vap88at/media/images/Learned%20Value%20Function_35147_11eb9195.png

[policy]: https://api.wandb.ai/files/hammania689/dqn/2vap88at/media/images/Learned%20Policy_35147_11eb9195.png
