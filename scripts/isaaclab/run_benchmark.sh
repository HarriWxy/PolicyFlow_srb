#!/bin/bash

# Franka
python scripts/isaaclab/train.py \
    --task=Isaac-Lift-Cube-Franka-PF \
    --headless \
    --log_dir=runs_pf/Franka_policyflow \
    --actor_hidden_dims 256 128 64 \
    --actor_activations elu elu elu linear \
    --critic_hidden_dims 256 128 64 \
    --critic_activations elu elu elu linear \
    --obs_embeding_dims 64 \
    --max_iterations=7001 \
    --rollouts=24

python scripts/isaaclab/train.py \
    --task=Isaac-Navigation-Flat-Anymal-C-PF \
    --headless \
    --log_dir=runs_pf/navigation_policyflow \
    --actor_hidden_dims 128 128\
    --actor_activations elu elu linear \
    --critic_hidden_dims 128 128 \
    --critic_activations elu elu linear \
    --obs_embeding_dims 64 \
    --max_iterations=7001 \
    --rollouts=8

# Quadcopter
python scripts/isaaclab/train.py \
    --task=Isaac-Quadcopter-Direct-PF \
    --headless \
    --log_dir=runs_pf/Quadcopter_policyflow \
    --actor_hidden_dims 64 64 \
    --actor_activations elu elu linear \
    --critic_hidden_dims 64 64 \
    --critic_activations elu elu linear \
    --obs_embeding_dims 64 \
    --max_iterations=7001 \
    --rollouts=24

# Anymal D
python scripts/isaaclab/train.py \
    --task=Isaac-Velocity-Flat-Anymal-D-PF \
    --headless \
    --log_dir=runs_pf/Anymal_d_policyflow \
    --actor_hidden_dims 128 128 128 \
    --actor_activations elu elu elu linear \
    --critic_hidden_dims 128 128 128 \
    --critic_activations elu elu elu linear \
    --obs_embeding_dims 64 \
    --max_iterations=7001 \
    --rollouts=24

# H1
python scripts/isaaclab/train.py \
    --task=Isaac-Velocity-Rough-H1-PF \
    --headless \
    --log_dir=runs_pf/H1_policyflow \
    --actor_hidden_dims 512 256 128 \
    --actor_activations elu elu elu linear \
    --critic_hidden_dims 512 256 128 \
    --critic_activations elu elu elu linear \
    --obs_embeding_dims 512 \
    --max_iterations=7001 \
    --rollouts=24

# Go2
python scripts/isaaclab/train.py \
    --task=Isaac-Velocity-Rough-Unitree-Go2-PF \
    --headless \
    --log_dir=runs_pf/Go2_policyflow \
    --actor_hidden_dims 512 256 128 \
    --actor_activations elu elu elu linear \
    --critic_hidden_dims 512 256 128 \
    --critic_activations elu elu elu linear \
    --obs_embeding_dims 512 \
    --max_iterations=7001 \
    --rollouts=24

# G1
python scripts/isaaclab/train.py \
    --task=Isaac-Velocity-Rough-G1-PF \
    --headless \
    --log_dir=runs_pf/G1_policyflow \
    --actor_hidden_dims 512 256 128 \
    --actor_activations elu elu elu linear \
    --critic_hidden_dims 512 256 128 \
    --critic_activations elu elu elu linear \
    --obs_embeding_dims 256 \
    --max_iterations=7001 \
    --rollouts=24

# Open-Drawer
python scripts/isaaclab/train.py \
    --task=Isaac-Open-Drawer-Franka-PF \
    --headless \
    --log_dir=runs_pf/Franka_open_drawer_policyflow \
    --actor_hidden_dims 256 128 64\
    --actor_activations elu elu elu linear \
    --critic_hidden_dims 256 128 64 \
    --critic_activations elu elu elu linear \
    --obs_embeding_dims 64 \
    --max_iterations=7001 \
    --rollouts=96


# play 
# python scripts/isaaclab/play.py \
#     --video --video_length 800 \
#     --task=Isaac-Lift-Cube-Franka-PF \
#     --headless \
#     --log_dir=runs_pf/Franka_policyflow \
#     --actor_hidden_dims 256 128 64 \
#     --actor_activations elu elu elu linear \
#     --critic_hidden_dims 256 128 64 \
#     --critic_activations elu elu elu linear \
#     --obs_embeding_dims 64 \
#     --max_iterations=7001 \
#     --rollouts=24