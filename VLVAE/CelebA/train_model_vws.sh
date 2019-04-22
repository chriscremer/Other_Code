



python3 train_CelebA.py --exp_name "test" \
								--input_size 64 \
								--joint_inf 1  --flow_int 1 --batch_size 20 \
								--w_logpy 100 --w_logpx .1 --max_beta 1 --z_size 50 \
								--data_dir "$HOME/celebA/celeb_data/" \
								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
								--just_classifier 0 \
								--train_classifier 0 \
								--which_gpu '0' \
								--quick_check 0 \
								--params_load_dir "$HOME/Documents/VLVAE_exps/celeba_3/params/" \
								--model_load_step 0 \
								--display_step 500 --trainingplot_steps 5000 --viz_steps 10000 \
								--start_storing_data_step 2001 --save_params_step 50000 \
								--textAR 1 \
								--classifier_load_params_dir "$HOME/celebA/classifier_params/" \
								--classifier_load_step 60000 \
								--max_steps 200000








