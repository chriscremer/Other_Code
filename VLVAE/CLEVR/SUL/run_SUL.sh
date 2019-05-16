



/home/ccremer/anaconda3/bin/python SUL.py --exp_name "test_may30_SUL_first2" \
								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
								--data_dir "$HOME/VL/data/clevr_single_object/"  \
								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
								--just_classifier 0 \
								--train_classifier 0 \
								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
								--classifier_load_step 150000 \
								--which_gpu '0' \
								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
								--model_load_step 0 \
								--display_step 500 --trainingplot_steps 4000 --viz_steps 10000 \
								--start_storing_data_step 1001 --save_params_step 30000 \
								--textAR 1 \
								--max_steps 400000 \
								--qy_detach 1
