



python3 SSL.py --exp_name "SSL_lower" \
								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
								--data_dir "$HOME/data/clevr_single_object/" \
								--save_to_dir "$HOME/results/VLVAE_exps/" \
								--just_classifier 0 \
								--train_classifier 1 \
								--classifier_load_params_dir "$HOME/data/clevr_two_objects/" \
								--classifier_load_step 0 \
								--which_gpu '1' \
								--quick_check 0 \
								--model_load_step 0 \
								--display_step 500 --trainingplot_steps 5000 --viz_steps 10000 \
								--start_storing_data_step 2001 --save_params_step 50000 \
								--ssl_type '0' \





