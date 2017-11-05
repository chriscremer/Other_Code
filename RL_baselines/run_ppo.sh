

# where is it saving?

# (cd $HOME/Documents/baselines/ && mpirun -np 8 python -m baselines.ppo1.run_atari)


export CUDA_VISIBLE_DEVICES=""

(cd $HOME/Documents/baselines/ && mpirun -np 2 python -m baselines.ppo1.run_atari)




