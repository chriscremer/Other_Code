

python3 train_jointVAE.py --exp_name "clevr_withpriorclassifier" \
								--multi 1  --joint_inf 0  --flow_int 1 --batch_size 20 \
								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
								--data_dir "$HOME/data/clevr_two_objects/" \
								--save_to_dir "$HOME/results/VLVAE_exps/" \
								--just_classifier 0 \
								--train_classifier 0 \
								--classifier_load_params_dir "$HOME/data/clevr_two_objects/" \
								--classifier_load_step 150000 \
								--which_gpu '0' \
								--quick_check 0 \
								--quick_check_data "$HOME/VL/two_objects_large/quick_stuff.pkl" \
								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_specific/params/" \
								--model_load_step 0 \
								--display_step 500 --trainingplot_steps 5000 --viz_steps 10000 \
								--start_storing_data_step 2001 --save_params_step 50000 \
								--train_prior_classifier 1 \






