#!/bin/bash

# vanilla dqn
python train_dqn.py 

# without target
python train_dqn.py --reset_interval=1 --tag="targetless"

# without replay memory
python train_dqn.py --replay_init=32  --bs=32 --tag="replayless"

# without replay and without target
python train_dqn.py --replay_init=32  --bs=32 --reset_interval=1 --tag="replayless targetless"
