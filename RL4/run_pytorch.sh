





export CUDA_VISIBLE_DEVICES="0"

# echo $(date +%Y%m%d)

date_time=`date '+%Y-%m-%d_%H-%M'`
save_dir=$HOME'/Documents/tmp/pytorch_pong_'$date_time
# save_dir=$HOME'/Documents/tmp/pytorch_pong_'

algo=a2c
n_processes=20
seed=1
env1="PongNoFrameskip-v4"
log_interval=20
save_interval=100000
n_frames=6000000
batch_size=100

# --vis-interval 100 
# echo $save_dir

# in parens so dir doesnt change
# (cd $HOME/Documents/pytorch-a2c-ppo-acktr/ && python main.py --env-name "PongNoFrameskip-v4" --num-processes $n_processes --save-dir $save_dir --log-dir $save_dir --log-interval 20 --save-interval 200 --vis-interval 100)
# (cd $HOME/Other_Code/RL4/pytorch-a2c-ppo-acktr/ && python main.py --env-name "PongNoFrameskip-v4" --num-processes $n_processes --save-dir $save_dir --log-dir $save_dir --log-interval 20 --save-interval 200 --vis-interval 100)


#Train modular version
# (cd $HOME/Other_Code/RL4/pytorch-a2c-ppo-acktr/ && python main_modular.py --env-name "PongNoFrameskip-v4" --num-processes $n_processes --save-dir $save_dir --log-dir $save_dir --log-interval 20 --save-interval 200 --vis-interval 100)
# (cd $HOME/Other_Code/RL4/pytorch-a2c-ppo-acktr/ && python main_modular2.py --env-name $env1 --num-processes $n_processes --save-dir $save_dir --log-dir $save_dir --log-interval $log_interval --save-interval $save_interval --seed $seed)
# (cd $HOME/Other_Code/RL4/pytorch-a2c-ppo-acktr/ && python main_modular_debug.py --env-name $env1 --num-processes $n_processes --save-dir $save_dir --log-dir $save_dir --log-interval $log_interval --save-interval $save_interval --seed $seed)






# #FOR PLOTTING
# to_load='pytorch_pong_2017-11-02_16-36'
# to_load=$HOME'/Documents/tmp/'$to_load

# (cd $HOME/Documents/pytorch-a2c-ppo-acktr/ && python visualize.py --dir $to_load)


# (cd $HOME/Documents/pytorch-a2c-ppo-acktr/ && python visualize.py)







algo=ppo
n_processes=20
seed=1
env1="PongNoFrameskip-v4"
log_interval=2
save_interval=100000
n_frames=6000000
batch_size=100
num_steps=200
load_path=$HOME'/Documents/tmp/pytorch_pong_2017-11-04_21-09/ppo/ppo.pt'

#original code
# (cd $HOME/Other_Code/RL4/pytorch-a2c-ppo-acktr/ && python main_ppo.py --env-name $env1 --algo $algo --use-gae --num-steps $num_steps --num-processes $n_processes --save-dir $save_dir --num-frames $n_frames --batch-size $batch_size --log-dir $save_dir --log-interval $log_interval --save-interval $save_interval --seed $seed)
#modular code
(cd $HOME/Other_Code/RL4/pytorch-a2c-ppo-acktr/ && python main_modular2.py --env-name $env1 --algo $algo --use-gae --num-steps $num_steps --num-processes $n_processes --save-dir $save_dir --load-path $load_path --num-frames $n_frames --batch-size $batch_size --log-interval $log_interval --save-interval $save_interval --seed $seed)








