#! /bin/bash


source activate test_env

python3 run_vae_cifar.py --exp_name "vae_test_q_flow_and_perm" \
								--z_size 384 --batch_size 64 \
								--enc_res_blocks 3 --dec_res_blocks 3  --n_prior_flows 5 \
								--which_gpu '0' \
								--data_dir "$HOME/Documents/" \
								--save_to_dir "$HOME/Documents/VAE2_exps/" \
								--display_step 500 --start_storing_data_step 2001 \
								--viz_steps 5000  --trainingplot_steps 5000 \
								--save_params_step 50000 --max_steps 400000 \
								--warmup_steps 20000 --continue_training 1 \
								--params_load_dir "$HOME/Documents/VAE2_exps/vae_test_prior_noperm/params/" \
								--model_load_step 0 \
								--save_output 1



# python3 run_vae_cifar.py --exp_name "vae_test_flow" \
# 								--z_size 384 --batch_size 64 \
# 								--enc_res_blocks 3 --dec_res_blocks 3  --n_prior_flows 5 \
# 								--which_gpu '0' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/VAE2_exps/" \
# 								--display_step 30 --start_storing_data_step 2001 \
# 								--viz_steps 3000  --trainingplot_steps 5000 \
# 								--save_params_step 50000 --max_steps 100000 \
# 								--warmup_steps 20000 --continue_training 1 \
# 								--params_load_dir "$HOME/Documents/VAE2_exps/vae_test_prior_noperm/params/" \
# 								--model_load_step 150000 \
# 								--to_stdout 1


# conda info --envs
# # source activate test_env
# # conda info --envs
# python3 run_vae_cifar.py --exp_name "vae_test_output" \
# 								--z_size 384 --batch_size 64 \
# 								--enc_res_blocks 3 --dec_res_blocks 3  --n_prior_flows 5 \
# 								--which_gpu '0' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/VAE2_exps/" \
# 								--display_step 30 --start_storing_data_step 2001 \
# 								--viz_steps 3000  --trainingplot_steps 5000 \
# 								--save_params_step 50000 --max_steps 100000 \
# 								--warmup_steps 20000 --continue_training 1 \
# 								--params_load_dir "$HOME/Documents/VAE2_exps/vae_test_fixedprior/params/" \
# 								--model_load_step 50000 \
# 								--to_stdout 0
# # source deactivate
# conda info --envs

# python3 run_vae_cifar.py --exp_name "vae_test_codecopy" \
# 								--z_size 384 --batch_size 64 \
# 								--enc_res_blocks 3 --dec_res_blocks 3  --n_prior_flows 5 \
# 								--which_gpu '0' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/VAE2_exps/" \
# 								--display_step 500 --start_storing_data_step 2001 \
# 								--viz_steps 5000  --trainingplot_steps 5000 \
# 								--save_params_step 50000 --max_steps 100000 \
# 								--warmup_steps 20000 --continue_training 0 \



# python3 run_vae_cifar.py --exp_name "vae_test_fixedprior" \
# 								--z_size 384 --batch_size 64 \
# 								--enc_res_blocks 3 --dec_res_blocks 3  --n_prior_flows 5 \
# 								--which_gpu '0' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/VAE2_exps/" \
# 								--display_step 500 --start_storing_data_step 2001 \
# 								--viz_steps 5000  --trainingplot_steps 5000 \
# 								--save_params_step 50000 --max_steps 100000 \
# 								--warmup_steps 20000 --continue_training 0 \




# python3 run_vae_cifar.py --exp_name "vae_test_actualflowq" \
# 								--z_size 384 --batch_size 64 \
# 								--enc_res_blocks 3 --dec_res_blocks 3  --n_prior_flows 5 \
# 								--which_gpu '0' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/VAE2_exps/" \
# 								--display_step 500 --start_storing_data_step 2001 \
# 								--viz_steps 5000  --trainingplot_steps 5000 \
# 								--save_params_step 50000 --max_steps 100000 \
# 								--warmup_steps 20000 --continue_training 0 \



# python3 run_vae_cifar.py --exp_name "vae_test_prior" \
# 								--z_size 384 --batch_size 64 \
# 								--which_gpu '0' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/VAE2_exps/" \
# 								--display_step 500 --start_storing_data_step 2001 \
# 								--viz_steps 5000  --trainingplot_steps 5000 \
# 								--save_params_step 50000 --max_steps 400000 \
# 								--warmup_steps 20000 --continue_training 0 \
# 								# --params_load_dir "$HOME/Documents/VAE2_exps/vae_z500_encdec32/params/" \
# 								# --model_load_step 50000 







# python3 run_vae_cifar.py --exp_name "vae_test_randomdeocder" \
# 								--z_size 384 --batch_size 64 \
# 								--which_gpu '0' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/VAE2_exps/" \
# 								--display_step 500 --start_storing_data_step 2001 \
# 								--viz_steps 5000  --trainingplot_steps 5000 \
# 								--save_params_step 50000 --max_steps 400000 \
# 								--warmup_steps 20000 --continue_training 0 \
# 								# --params_load_dir "$HOME/Documents/VAE2_exps/vae_z500_encdec32/params/" \
# 								# --model_load_step 50000 






# python3 run_vae_cifar.py --exp_name "vae_test_minmax" \
# 								--z_size 384 --batch_size 64 \
# 								--which_gpu '0' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/VAE2_exps/" \
# 								--display_step 500 --start_storing_data_step 2001 \
# 								--viz_steps 5000  --trainingplot_steps 5000 \
# 								--save_params_step 50000 --max_steps 400000 \
# 								--warmup_steps 20000 --continue_training 0 \
# 								# --params_load_dir "$HOME/Documents/VAE2_exps/vae_z500_encdec32/params/" \
# 								# --model_load_step 50000 





# python3 run_vae_cifar.py --exp_name "vae_test_grid_withz_prior" \
# 								--z_size 384 --batch_size 64 \
# 								--which_gpu '0' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/VAE2_exps/" \
# 								--display_step 500 --start_storing_data_step 2001 \
# 								--viz_steps 50  --trainingplot_steps 5000 \
# 								--save_params_step 50000 --max_steps 400000 \
# 								--warmup_steps 20000 --continue_training 0 \
# 								# --params_load_dir "$HOME/Documents/VAE2_exps/vae_z500_encdec32/params/" \
# 								# --model_load_step 50000 




# python3 run_vae_cifar.py --exp_name "train_bpd" \
# 								--z_size 200 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/train_withmultipleinfnets/params/" \
# 								--model_load_step 250000 






# python3 run_vae_cifar.py --exp_name "train_withmultipleinfnets" \
# 								--z_size 200 \
# 								--which_gpu '0' \
# 								# --params_load_dir "$HOME/Documents/first/params/" \
# 								# --model_load_step 100000 





# python3 run_vae_cifar.py --exp_name "first_newinfnet" \
# 								--z_size 200 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/first/params/" \
# 								--model_load_step 100000 




# python3 run_vae_cifar.py --exp_name "train_infnet" \
# 								--z_size 200 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/first/params/" \
# 								--model_load_step 100000 






# python3 run_vae_cifar.py --exp_name first --z_size 200



# python3 run_vae_cifar.py --exp_name "first" \
# 								--z_size 200 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/first/params/" \
# 								--model_load_step 100000 


								# --multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
								# --w_logpy 1 --w_logpx 1 --max_beta 1 --z_size 2 \
								# --data_dir "$HOME/VL/data/clevr_single_object/" \
								# --save_to_dir "$HOME/Documents/VLVAE_exps/" \
								# --just_classifier 0 \
								# --train_classifier 0 \
								# --classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
								# --classifier_load_step 0 \
								# --which_gpu '1' \
								# --params_load_dir "$HOME/Documents/VLVAE_exps/train_2D_adaptiveB_3/params/" \
								# --model_load_step 60000 \
								# --display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
								# --start_storing_data_step 2001 --save_params_step 50000 \
								# --ssl_type '0' \
								# --textAR 1 \
								# --max_steps 300000






