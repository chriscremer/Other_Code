




# Run deepq

#Run
# python -m baselines.deepq.experiments.atari.train --env Pong --save-dir $HOME/Documents/tmp/baselines_pong2
# python $HOME/Documents/baselines/baselines/deepq/experiments/atari/train.py --env Pong --save-dir $HOME/Documents/tmp/baselines_pong2


export CUDA_VISIBLE_DEVICES="0"

# echo $(date +%Y%m%d)

date_time=`date '+%Y-%m-%d_%H-%M'`
save_dir=$HOME'/Documents/tmp/baselines_pong_'$date_time


# in parens so dir doesnt change
(cd $HOME/Documents/baselines/ && python -m baselines.deepq.experiments.atari.train --env Pong --save-dir $save_dir)
# (cd $HOME/Documents/baselines/ && python -m baselines.deepq.experiments.atari.train --env Pong --save-dir $HOME/Documents/tmp/baselines_pong3 --gym-monitor)

# to see it play
# (cd $HOME/Documents/baselines/ && python -m baselines.deepq.experiments.atari.enjoy --env Pong --model-dir $HOME/Documents/tmp/baselines_pong2/model-1000000)

