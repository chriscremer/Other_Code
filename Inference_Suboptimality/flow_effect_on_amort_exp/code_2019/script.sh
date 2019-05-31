



# /home/ccremer/anaconda3/bin/python compute_gaps.py --exp_name "fashion_flow_cond_10_encoderonly" \
# 								--x_size 784 --z_size 20  \
# 								--q_dist 'Flow_Cond' \
# 								--n_flows 10 \
# 								--which_gpu '1' \
# 								--save_to_dir "$HOME/Documents/Inf_Sub/" \
# 								--display_step 1 \
# 								--generator_params_load_dir "$HOME/Documents/Inf_Sub/fashion_gauss/params/" \
# 								--generator_load_step 200000 \
# 								--encoder_params_load_dir "$HOME/Documents/Inf_Sub/fashion_flow_cond_10_encoderonly/params/" \
# 								--encoder_load_step 200000 \



# /home/ccremer/anaconda3/bin/python compute_gaps.py --exp_name "fashion_flow_cond_encoderonly" \
# 								--x_size 784 --z_size 20  \
# 								--q_dist 'Flow_Cond' \
# 								--n_flows 5 \
# 								--which_gpu '1' \
# 								--save_to_dir "$HOME/Documents/Inf_Sub/" \
# 								--display_step 1 \
# 								--generator_params_load_dir "$HOME/Documents/Inf_Sub/fashion_gauss/params/" \
# 								--generator_load_step 200000 \
# 								--encoder_params_load_dir "$HOME/Documents/Inf_Sub/fashion_flow_cond_encoderonly/params/" \
# 								--encoder_load_step 200000 \



/home/ccremer/anaconda3/bin/python compute_gaps.py --exp_name "fashion_gauss_encoderonly" \
								--x_size 784 --z_size 20  \
								--q_dist 'Gauss' \
								--n_flows 2 \
								--which_gpu '1' \
								--save_to_dir "$HOME/Documents/Inf_Sub/" \
								--display_step 1 \
								--generator_params_load_dir "$HOME/Documents/Inf_Sub/fashion_gauss/params/" \
								--generator_load_step 200000 \
								--encoder_params_load_dir "$HOME/Documents/Inf_Sub/fashion_gauss_encoderonly/params/" \
								--encoder_load_step 200000 \




# /home/ccremer/anaconda3/bin/python train.py --exp_name "fashion_flow_cond_10_encoderonly" \
# 								--x_size 784 --z_size 20 --batch_size 64 \
# 								--q_dist 'Flow_Cond' \
# 								--n_flows 10 \
# 								--which_gpu '1' \
# 								--save_to_dir "$HOME/Documents/Inf_Sub/" \
# 								--display_step 500 --start_storing_data_step 501 \
# 								--trainingplot_steps 10000 --viz_steps 10000 \
# 								--save_params_step 50000 --max_steps 200000 \
# 								--warmup_steps 1 --continue_training 0 \
# 								--generator_params_load_dir "$HOME/Documents/Inf_Sub/fashion_gauss/params/" \
# 								--generator_load_step 200000 \
# 								--train_encoder_only 1 \






# /home/ccremer/anaconda3/bin/python compute_gaps.py --exp_name "fashion_flow_10_encoderonly" \
# 								--x_size 784 --z_size 20  \
# 								--q_dist 'Flow' \
# 								--n_flows 10 \
# 								--which_gpu '0' \
# 								--save_to_dir "$HOME/Documents/Inf_Sub/" \
# 								--display_step 1 \
# 								--generator_params_load_dir "$HOME/Documents/Inf_Sub/fashion_gauss/params/" \
# 								--generator_load_step 200000 \
# 								--encoder_params_load_dir "$HOME/Documents/Inf_Sub/fashion_flow_10_encoderonly/params/" \
# 								--encoder_load_step 200000 \








# /home/ccremer/anaconda3/bin/python train.py --exp_name "fashion_flow_cond_encoderonly" \
# 								--x_size 784 --z_size 20 --batch_size 64 \
# 								--q_dist 'Flow_Cond' \
# 								--n_flows 5 \
# 								--which_gpu '1' \
# 								--save_to_dir "$HOME/Documents/Inf_Sub/" \
# 								--display_step 500 --start_storing_data_step 501 \
# 								--trainingplot_steps 10000 --viz_steps 10000 \
# 								--save_params_step 50000 --max_steps 200000 \
# 								--warmup_steps 1 --continue_training 0 \
# 								--generator_params_load_dir "$HOME/Documents/Inf_Sub/fashion_gauss/params/" \
# 								--generator_load_step 200000 \
# 								--train_encoder_only 1 \










# /home/ccremer/anaconda3/bin/python compute_gaps.py --exp_name "fashion_flow_2_encoderonly" \
# 								--x_size 784 --z_size 20  \
# 								--q_dist 'Flow' \
# 								--n_flows 2 \
# 								--which_gpu '0' \
# 								--save_to_dir "$HOME/Documents/Inf_Sub/" \
# 								--display_step 1 \
# 								--generator_params_load_dir "$HOME/Documents/Inf_Sub/fashion_gauss/params/" \
# 								--generator_load_step 200000 \
# 								--encoder_params_load_dir "$HOME/Documents/Inf_Sub/fashion_flow_2_encoderonly/params/" \
# 								--encoder_load_step 200000 \







# /home/ccremer/anaconda3/bin/python train.py --exp_name "fashion_flow_2_encoderonly" \
# 								--x_size 784 --z_size 20 --batch_size 64 \
# 								--q_dist 'Flow' \
# 								--n_flows 2 \
# 								--which_gpu '1' \
# 								--save_to_dir "$HOME/Documents/Inf_Sub/" \
# 								--display_step 500 --start_storing_data_step 501 \
# 								--trainingplot_steps 10000 --viz_steps 10000 \
# 								--save_params_step 50000 --max_steps 200000 \
# 								--warmup_steps 1 --continue_training 0 \
# 								--generator_params_load_dir "$HOME/Documents/Inf_Sub/fashion_gauss/params/" \
# 								--generator_load_step 200000 \
# 								--train_encoder_only 1 \



# /home/ccremer/anaconda3/bin/python train.py --exp_name "fashion_gauss_encoderonly" \
# 								--x_size 784 --z_size 20 --batch_size 64 \
# 								--q_dist 'Gauss' \
# 								--which_gpu '1' \
# 								--save_to_dir "$HOME/Documents/Inf_Sub/" \
# 								--display_step 500 --start_storing_data_step 501 \
# 								--trainingplot_steps 10000 --viz_steps 10000 \
# 								--save_params_step 50000 --max_steps 200000 \
# 								--warmup_steps 1 --continue_training 0 \
# 								--generator_params_load_dir "$HOME/Documents/Inf_Sub/fashion_gauss/params/" \
# 								--generator_load_step 200000 \
# 								--train_encoder_only 1 \






# /home/ccremer/anaconda3/bin/python train.py --exp_name "fashion_flow_10_encoderonly" \
# 								--x_size 784 --z_size 20 --batch_size 64 \
# 								--q_dist 'Flow' \
# 								--n_flows 10 \
# 								--which_gpu '1' \
# 								--save_to_dir "$HOME/Documents/Inf_Sub/" \
# 								--display_step 500 --start_storing_data_step 501 \
# 								--trainingplot_steps 10000 --viz_steps 10000 \
# 								--save_params_step 50000 --max_steps 200000 \
# 								--warmup_steps 1 --continue_training 0 \
# 								--generator_params_load_dir "$HOME/Documents/Inf_Sub/fashion_gauss/params/" \
# 								--generator_load_step 200000 \
# 								--train_encoder_only 1 \














# /home/ccremer/anaconda3/bin/python compute_gaps.py --exp_name "fashion_gauss" \
# 								--x_size 784 --z_size 20  \
# 								--which_gpu '1' \
# 								--q_dist 'Gauss' \
# 								--save_to_dir "$HOME/Documents/Inf_Sub/" \
# 								--display_step 1 \
# 								--generator_params_load_dir "$HOME/Documents/Inf_Sub/vae_gauss/params/" \
# 								--generator_load_step 100000 \
# 								--encoder_params_load_dir "$HOME/Documents/Inf_Sub/vae_encoder_only_gauss/params/" \
# 								--encoder_load_step 100000 \









# /home/ccremer/anaconda3/bin/python train.py --exp_name "fashion_flow_2" \
# 								--x_size 784 --z_size 20 --batch_size 64 \
# 								--q_dist 'Flow' \
# 								--n_flows 2 \
# 								--which_gpu '1' \
# 								--save_to_dir "$HOME/Documents/Inf_Sub/" \
# 								--display_step 500 --start_storing_data_step 501 \
# 								--trainingplot_steps 10000 --viz_steps 10000 \
# 								--save_params_step 50000 --max_steps 200000 \
# 								--warmup_steps 1 --continue_training 0 \
# 								# --generator_params_load_dir "$HOME/Documents/Inf_Sub/vae_gauss/params/" \
# 								# --generator_load_step 100000 \
# 								# --train_encoder_only 1 \



# /home/ccremer/anaconda3/bin/python train.py --exp_name "fashion_gauss" \
# 								--x_size 784 --z_size 20 --batch_size 64 \
# 								--q_dist 'Gauss' \
# 								--which_gpu '0' \
# 								--save_to_dir "$HOME/Documents/Inf_Sub/" \
# 								--display_step 500 --start_storing_data_step 501 \
# 								--trainingplot_steps 10000 --viz_steps 10000 \
# 								--save_params_step 50000 --max_steps 200000 \
# 								--warmup_steps 1 --continue_training 0 \
# 								# --generator_params_load_dir "$HOME/Documents/Inf_Sub/vae_gauss/params/" \
# 								# --generator_load_step 100000 \
# 								# --train_encoder_only 1 \






# /home/ccremer/anaconda3/bin/python train.py --exp_name "fashion_flow_10" \
# 								--x_size 784 --z_size 20 --batch_size 64 \
# 								--q_dist 'Flow' \
# 								--n_flows 10 \
# 								--which_gpu '1' \
# 								--save_to_dir "$HOME/Documents/Inf_Sub/" \
# 								--display_step 500 --start_storing_data_step 501 \
# 								--trainingplot_steps 10000 --viz_steps 10000 \
# 								--save_params_step 50000 --max_steps 200000 \
# 								--warmup_steps 1 --continue_training 0 \
# 								# --generator_params_load_dir "$HOME/Documents/Inf_Sub/vae_gauss/params/" \
# 								# --generator_load_step 100000 \
# 								# --train_encoder_only 1 \
















# /home/ccremer/anaconda3/bin/python compute_gaps.py --exp_name "vae_encoder_only_gauss" \
# 								--x_size 784 --z_size 20  \
# 								--which_gpu '1' \
# 								--q_dist 'Gauss' \
# 								--save_to_dir "$HOME/Documents/Inf_Sub/" \
# 								--display_step 1 \
# 								--generator_params_load_dir "$HOME/Documents/Inf_Sub/vae_gauss/params/" \
# 								--generator_load_step 100000 \
# 								--encoder_params_load_dir "$HOME/Documents/Inf_Sub/vae_encoder_only_gauss/params/" \
# 								--encoder_load_step 100000 \





# /home/ccremer/anaconda3/bin/python compute_gaps.py --exp_name "vae_encoder_only_flow2" \
# 								--x_size 784 --z_size 20  \
# 								--which_gpu '1' \
# 								--q_dist 'Flow' \
# 								--n_flows 2 \
# 								--save_to_dir "$HOME/Documents/Inf_Sub/" \
# 								--display_step 1 \
# 								--generator_params_load_dir "$HOME/Documents/Inf_Sub/vae_gauss/params/" \
# 								--generator_load_step 100000 \
# 								--encoder_params_load_dir "$HOME/Documents/Inf_Sub/vae_encoder_only_flow2/params/" \
# 								--encoder_load_step 100000 \






# /home/ccremer/anaconda3/bin/python compute_gaps.py --exp_name "vae_encoder_only_flow4" \
# 								--x_size 784 --z_size 20  \
# 								--which_gpu '1' \
# 								--q_dist 'Flow' \
# 								--n_flows 4 \
# 								--save_to_dir "$HOME/Documents/Inf_Sub/" \
# 								--display_step 1 \
# 								--generator_params_load_dir "$HOME/Documents/Inf_Sub/vae_gauss/params/" \
# 								--generator_load_step 100000 \
# 								--encoder_params_load_dir "$HOME/Documents/Inf_Sub/vae_encoder_only_flow4/params/" \
# 								--encoder_load_step 100000 \





# /home/ccremer/anaconda3/bin/python compute_gaps.py --exp_name "vae_encoder_only_flow5" \
# 								--x_size 784 --z_size 20  \
# 								--which_gpu '1' \
# 								--q_dist 'Flow' \
# 								--n_flows 5 \
# 								--save_to_dir "$HOME/Documents/Inf_Sub/" \
# 								--display_step 1 \
# 								--generator_params_load_dir "$HOME/Documents/Inf_Sub/vae_gauss/params/" \
# 								--generator_load_step 100000 \
# 								--encoder_params_load_dir "$HOME/Documents/Inf_Sub/vae_encoder_only_flow5/params/" \
# 								--encoder_load_step 200000 \




# /home/ccremer/anaconda3/bin/python compute_gaps.py --exp_name "vae_encoder_only_flow10" \
# 								--x_size 784 --z_size 20  \
# 								--which_gpu '1' \
# 								--q_dist 'Flow' \
# 								--n_flows 10 \
# 								--save_to_dir "$HOME/Documents/Inf_Sub/" \
# 								--display_step 1 \
# 								--generator_params_load_dir "$HOME/Documents/Inf_Sub/vae_gauss/params/" \
# 								--generator_load_step 100000 \
# 								--encoder_params_load_dir "$HOME/Documents/Inf_Sub/vae_encoder_only_flow10/params/" \
# 								--encoder_load_step 200000 \











# /home/ccremer/anaconda3/bin/python train.py --exp_name "vae_encoder_only_flow10" \
# 								--x_size 784 --z_size 20 --batch_size 64 \
# 								--q_dist 'Flow' \
# 								--n_flows 10 \
# 								--which_gpu '1' \
# 								--save_to_dir "$HOME/Documents/Inf_Sub/" \
# 								--display_step 500 --start_storing_data_step 1 \
# 								--trainingplot_steps 10000 --viz_steps 10000 \
# 								--save_params_step 50000 --max_steps 200000 \
# 								--warmup_steps 1 --continue_training 0 \
# 								--generator_params_load_dir "$HOME/Documents/Inf_Sub/vae_gauss/params/" \
# 								--generator_load_step 100000 \
# 								--train_encoder_only 1 \








# /home/ccremer/anaconda3/bin/python train.py --exp_name "vae_encoder_only_flow5" \
# 								--x_size 784 --z_size 20 --batch_size 64 \
# 								--q_dist 'Flow' \
# 								--n_flows 5 \
# 								--which_gpu '0' \
# 								--save_to_dir "$HOME/Documents/Inf_Sub/" \
# 								--display_step 500 --start_storing_data_step 1 \
# 								--trainingplot_steps 10000 --viz_steps 10000 \
# 								--save_params_step 50000 --max_steps 200000 \
# 								--warmup_steps 1 --continue_training 0 \
# 								--generator_params_load_dir "$HOME/Documents/Inf_Sub/vae_gauss/params/" \
# 								--generator_load_step 100000 \
# 								--train_encoder_only 1 \














# /home/ccremer/anaconda3/bin/python train.py --exp_name "vae_encoder_only_flow4" \
# 								--x_size 784 --z_size 20 --batch_size 64 \
# 								--q_dist 'Flow' \
# 								--n_flows 4 \
# 								--which_gpu '1' \
# 								--save_to_dir "$HOME/Documents/Inf_Sub/" \
# 								--display_step 500 --start_storing_data_step 1 \
# 								--trainingplot_steps 10000 --viz_steps 10000 \
# 								--save_params_step 50000 --max_steps 100000 \
# 								--warmup_steps 1 --continue_training 0 \
# 								--generator_params_load_dir "$HOME/Documents/Inf_Sub/vae_gauss/params/" \
# 								--generator_load_step 100000 \
# 								--train_encoder_only 1 \







# /home/ccremer/anaconda3/bin/python train.py --exp_name "vae_encoder_only_flow2" \
# 								--x_size 784 --z_size 20 --batch_size 64 \
# 								--q_dist 'Flow' \
# 								--n_flows 2 \
# 								--which_gpu '1' \
# 								--save_to_dir "$HOME/Documents/Inf_Sub/" \
# 								--display_step 500 --start_storing_data_step 1 \
# 								--trainingplot_steps 10000 --viz_steps 10000 \
# 								--save_params_step 50000 --max_steps 100000 \
# 								--warmup_steps 1 --continue_training 0 \
# 								--generator_params_load_dir "$HOME/Documents/Inf_Sub/vae_gauss/params/" \
# 								--generator_load_step 100000 \
# 								--train_encoder_only 1 \








# /home/ccremer/anaconda3/bin/python train.py --exp_name "vae_encoder_only_gauss" \
# 								--x_size 784 --z_size 20 --batch_size 64 \
# 								--q_dist 'Gauss' \
# 								--which_gpu '0' \
# 								--save_to_dir "$HOME/Documents/Inf_Sub/" \
# 								--display_step 500 --start_storing_data_step 1 \
# 								--trainingplot_steps 10000 --viz_steps 10000 \
# 								--save_params_step 50000 --max_steps 100000 \
# 								--warmup_steps 1 --continue_training 0 \
# 								--generator_params_load_dir "$HOME/Documents/Inf_Sub/vae_gauss/params/" \
# 								--generator_load_step 100000 \
# 								--train_encoder_only 1 \








# /home/ccremer/anaconda3/bin/python compute_gaps.py --exp_name "vae_gauss" \
# 								--x_size 784 --z_size 20  \
# 								--which_gpu '0' \
# 								--q_dist 'Gauss' \
# 								--save_to_dir "$HOME/Documents/Inf_Sub/" \
# 								--display_step 1 \
# 								--params_load_dir "$HOME/Documents/Inf_Sub/vae_gauss/params/" \
# 								--load_step 100000 \




# /home/ccremer/anaconda3/bin/python compute_gaps.py --exp_name "test" \
# 								--x_size 784 --z_size 20  \
# 								--which_gpu '1' \
# 								--save_to_dir "$HOME/Documents/Inf_Sub/" \
# 								--display_step 1 \
# 								--params_load_dir "$HOME/Documents/Inf_Sub/test/params/" \
# 								--load_step 100000 \





# /home/ccremer/anaconda3/bin/python train.py --exp_name "vae_gauss" \
# 								--x_size 784 --z_size 20 --batch_size 64 \
# 								--q_dist 'Gauss' \
# 								--which_gpu '0' \
# 								--save_to_dir "$HOME/Documents/Inf_Sub/" \
# 								--display_step 500 --start_storing_data_step 1 \
# 								--trainingplot_steps 10000 --viz_steps 10000 \
# 								--save_params_step 20000 --max_steps 100000 \
# 								--warmup_steps 1 --continue_training 0 \






