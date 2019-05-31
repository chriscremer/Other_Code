



/home/ccremer/anaconda3/bin/python train.py --exp_name "siva_test3_warmup" \
								--img_dim 112 --z_size 100 --batch_size 64 \
								--q_dist 'Gauss' \
								--which_gpu '0' \
								--save_to_dir "$HOME/Documents/SIVA/" \
								--display_step 500 --start_storing_data_step 400 \
								--trainingplot_steps 5000 --viz_steps 10000 \
								--save_params_step 100000 --max_steps 200000 \
								--warmup_steps 10000 --continue_training 0 \
								# --generator_params_load_dir "$HOME/Documents/Inf_Sub/fashion_gauss/params/" \
								# --generator_load_step 200000 \
								# --train_encoder_only 1 \



