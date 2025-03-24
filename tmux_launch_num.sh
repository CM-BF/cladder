#!/bin/bash

# Usage: GPU_IDX=1,2,3,4,5 bash ./tmux_launch.sh CSR sweep_id [count]

# Define the name of the tmux session
IFS=',' read -r -a gpu_idx <<< $GPU_IDX
conda_env=$1
sweep_id=$2
count=$3
session_name="exps_${sweep_id}"

# Check if the session name already exists, and if so, increment it
suffix=1
original_session_name=$session_name
while tmux has-session -t $session_name 2>/dev/null; do
    session_name="${original_session_name}_${suffix}"
    ((suffix++))
done

# Start a new tmux session
tmux new-session -d -s $session_name

# Create windows for each agent
for i in $(seq 1 ${#gpu_idx[@]}); do
    if [ $i -eq 1 ]; then
        # For the first agent, rename the first window rather than creating a new one
        tmux rename-window -t $session_name:1 "Agent $i"
        tmux send-keys -t $session_name:1 "conda activate ${conda_env}" C-m
        if [ -n "$count" ]; then
            tmux send-keys -t $session_name:1 "CUDA_VISIBLE_DEVICES=${gpu_idx[i-1]} wandb agent --count ${count} ${sweep_id}" C-m
        else
            tmux send-keys -t $session_name:1 "CUDA_VISIBLE_DEVICES=${gpu_idx[i-1]} wandb agent ${sweep_id}" C-m
        fi
    else
        # For other agents, create new windows
        tmux new-window -t $session_name -n "Agent $i"
        tmux send-keys -t $session_name:"Agent $i" "conda activate ${conda_env}" C-m
        if [ -n "$count" ]; then
            tmux send-keys -t $session_name:"Agent $i" "CUDA_VISIBLE_DEVICES=${gpu_idx[i-1]} wandb agent --count ${count} ${sweep_id}" C-m
        else
            tmux send-keys -t $session_name:"Agent $i" "CUDA_VISIBLE_DEVICES=${gpu_idx[i-1]} wandb agent ${sweep_id}" C-m
        fi
    fi
done

# Attach to the tmux session (optional)
# tmux attach-session -t $session_name