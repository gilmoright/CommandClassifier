#!/bin/bash

#SBATCH -J tune_rfc_etc
#SBATCH -D /s/ls4/users/grartem/RL_robots/CommandClassifier
#SBATCH -o /s/ls4/users/grartem/RL_robots/CommandClassifier/Logs/tune_%x_%j.out
#SBATCH -e /s/ls4/users/grartem/RL_robots/CommandClassifier/Logs/tune_%x_%j.err
#SBATCH -p hpc5-el7-gpu-3d
#SBATCH -n 16
#SBATCH --gres=gpu:k80:1
#SBATCH --time=24:00:00

export HOME=/s/ls4/users/grartem
export PATH=$HOME/anaconda3/envs/rl_robots/bin:$PATH

export LD_LIBRARY_PATH=/s/ls4/sw/cuda/10.1/lib64:/s/ls4/sw/cuda/10.1/nvvm/lib64:$HOME/installation_dists/cudnn-9.0-linux-x64-v7.1.ga/lib64:
# RandomForestClassifier ExtraTreesClassifier ExtraTreeClassifier DecisionTreeClassifier KNeighborsClassifier RadiusNeighborsClassifier
python RobotCommandClassifier/ray_tune_cls.py --algo_class RandomForestClassifier