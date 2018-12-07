



/home/ccremer/anaconda3/bin/python run_vae_cifar.py --exp_name "vae_z500" \
								--z_size 500 --batch_size 64 \
								--which_gpu '0' \
								--data_dir "$HOME/Documents/" \
								--save_to_dir "$HOME/Documents/VAE2_exps/" \
								--display_step 500 --start_storing_data_step 2001 \
								--viz_steps 5000  --trainingplot_steps 5000 \
								--save_params_step 50000 --max_steps 400000 
								# --params_load_dir "$HOME/Documents/train_withmultipleinfnets/params/" \
								# --model_load_step 250000 










# /home/ccremer/anaconda3/bin/python train_qy_not_prior.py --exp_name "test_models" \
# 								--input_size 64 \
# 								--joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 1 --w_logpx 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/celebA/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--which_gpu '1' \
# 								--quick_check 0 \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/celeba_3/params/" \
# 								--model_load_step 250000 \
# 								--display_step 500 --trainingplot_steps 5000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/celeba_classifier/params" \
# 								--classifier_load_step 60000 \
# 								--max_steps 400000 \
# 								--qy_type 'true' \
# 								--learn_prior 0



