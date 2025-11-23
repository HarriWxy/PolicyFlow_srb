#!/bin/bash
n=5  

for ((i=1; i<=n; i++))
do
    echo "Running iteration $i"
    python scripts/isaaclab/train.py \
        --task=Isaac-Lift-Cube-Franka-PF \
        --headless \
        --log_dir=runs/Franka_reflow \
        --actor_hidden_dims 256 128 64 \
        --actor_activations elu elu elu linear \
        --critic_hidden_dims 256 128 64 \
        --critic_activations elu elu elu linear \
        --obs_embeding_dims 64 \
        --max_iterations=7001 \
        --rollouts=24
done