# How to use jupyter lab in IFL cluster?

1. connect to the VPN:

(optional) if you don't want to input the username and password, you can create a file called passwd_ifl. The first row of it your username while the second row is the password. Then run:
```
sudo openvpn --config ~/openvpn/IFL.ovpn --auth-user-pass ~/openvpn/passwd_ifl
```

2. connect to the cluster with:
```
ssh -L 8888:localhost:8888 usr_mlmi@10.23.0.18
```

3. (option) use tmux to keep the command running
```
tmux
```
you can detach it with `Ctrl + B` then press `D` and attach it by 
```
tmux attach -s <session num>
```

some useful command for tmux:
```
tmux split-window -h
tmux set mouse on
```
use `Ctrl B` then `?` for more infomation

4. use `srun` to submit a job:
```
srun --time=0-12:00:00 --gres=gpu:1 --cpus-per-task=6 --mem=24G --pty bash
```

5. the current terminal should be in the computing node of cluster, run:
```
ml cuda
ml miniconda3
conda activate test
ssh -N -f -R 8888:localhost:8888 10.23.0.18
jupyter-lab --port 8888
```

6. click the URL to open the juputer lab!