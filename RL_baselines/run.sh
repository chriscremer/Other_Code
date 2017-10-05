



#Run
# python -m baselines.deepq.experiments.atari.train --env Pong --save-dir $HOME/Documents/tmp/baselines_pong2
# python $HOME/Documents/baselines/baselines/deepq/experiments/atari/train.py --env Pong --save-dir $HOME/Documents/tmp/baselines_pong2




# in parens so dir doesnt change
# (cd $HOME/Documents/baselines/ && python -m baselines.deepq.experiments.atari.train --env Pong --save-dir $HOME/Documents/tmp/baselines_pong2)


# to see it play
(cd $HOME/Documents/baselines/ && python -m baselines.deepq.experiments.atari.enjoy --env Pong --model-dir $HOME/Documents/tmp/baselines_pong2/model-1000000)



