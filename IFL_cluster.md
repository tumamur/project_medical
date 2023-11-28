# How to use jupyter lab in IFL cluster?

1. connect to the VPN:

2. connect to the cluster with:
```
ssh -L 8888:localhost:8888 usr_mlmi@10.23.0.18
```

3. use `srun` to submit a job:
```
srun --time=0-12:00:00 --gres=gpu:1 --cpus-per-task=6 --mem=24G --pty bash
```

4. the current terminal should be in the computing node of cluster, run:
```
ml cuda
ml miniconda3
conda activate test
ssh -N -f -R 8888:localhost:8888 10.23.0.18
jupyter-lab --port 8888
```

5. click the URL to open the juputer lab!


## to do

jupyter lab password

tmux

vpn password

ssh-key

sbatch

env in login node or computing node