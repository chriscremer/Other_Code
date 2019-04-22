


/home/ccremer/anaconda3/bin/python SSL.py --exp_name "SSL_oneobject_marg_mid" \
								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
								--w_logpy 100 --w_logpx .01 --max_beta 1 --z_size 50 \
								--data_dir "$HOME/VL/data/clevr_single_object/" \
								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
								--just_classifier 0 \
								--train_classifier 0 \
								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
								--classifier_load_step 150000 \
								--which_gpu '0' \
								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_oneobject_marg_mid/params/" \
								--model_load_step 150000 \
								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
								--start_storing_data_step 2001 --save_params_step 50000 \
								--ssl_type '1' \
								--textAR 1 \
								--ssl_percent 1\




# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "SSL_oneobject_marg_mid_3" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 1000 --w_logpx .1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_oneobject_marg_mid_3/params/" \
# 								--model_load_step 150000 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '1' \
# 								--textAR 1 \
# 								--ssl_percent 1\



# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "SSL_oneobject_marg_mid_2" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_oneobject_marg_mid_2/params/" \
# 								--model_load_step 150000 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '1' \
# 								--textAR 1 \
# 								--ssl_percent 1\




# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "SSL_oneobject_marg_mid_4" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 500 --w_logpx .05 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_oneobject_marg_mid_4/params/" \
# 								--model_load_step 150000 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '1' \
# 								--textAR 1 \
# 								--max_steps 300000 \
# 								--ssl_percent 1\




# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "SSL_oneobject_marg_mid_5" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 50 --w_logpx .005 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_oneobject_marg_mid_5/params/" \
# 								--model_load_step 150000 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '1' \
# 								--textAR 1 \
# 								--max_steps 300000 \
# 								--ssl_percent 1\





# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "SSL_oneobject_marg_mid_4" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 500 --w_logpx .05 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_oneobject_marg_mid_4/params/" \
# 								--model_load_step 1 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '1' \
# 								--textAR 1 \
# 								--max_steps 300000 \










# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "SSL_oneobject_marg_mid_4" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 500 --w_logpx .05 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_specific/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '1' \
# 								--textAR 1 \
# 								--max_steps 300000 \














# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "SSL_get_percent" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 400 --w_logpx .04 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_oneobject_marg_lower/params/" \
# 								--model_load_step 150000 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '0' \
# 								--textAR 1 \









# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "SSL_get_percent" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 100 --w_logpx .01 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_oneobject_marg_mid/params/" \
# 								--model_load_step 150000 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '1' \
# 								--textAR 1 \



# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "SSL_get_percent" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_oneobject_marg_mid_2/params/" \
# 								--model_load_step 150000 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '1' \
# 								--textAR 1 \










# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "SSL_oneobject_marg_lower" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 400 --w_logpx .04 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_specific/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '0' \
# 								--textAR 1 \





# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "SSL_get_percent" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_oneobject_marg_mid_2/params/" \
# 								--model_load_step 150000 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '1' \
# 								--textAR 1 \






# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "SSL_oneobject_marg_mid_3" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 1000 --w_logpx .1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_specific/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '1' \
# 								--textAR 1 \



# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "SSL_oneobject_marg_upper" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 400 --w_logpx .04 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_specific/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '2' \
# 								--textAR 1 \












# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "SUL_marg_twoobjects_upper2" \
# 								--multi 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_specific/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '2' \
# 								--textAR 1 \




# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "SUL_joint_twoobjects_upper" \
# 								--multi 1 --joint_inf 1  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_specific/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '2' \
# 								--textAR 1 \





# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "SUL_marg_twoobjects_lower" \
# 								--multi 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_specific/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '0' \
# 								--textAR 1 \





# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "SUL_marg_twoobjects_mid" \
# 								--multi 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_specific/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '1' \
# 								--textAR 1 \






# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "SSL_joint_mid2" \
# 								--multi 0 --singlev2 1 --joint_inf 1  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_specific/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '1' \
# 								--textAR 1 \




# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "SSL_joint_lower2" \
# 								--multi 0 --singlev2 1 --joint_inf 1  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_specific/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '0' \
# 								--textAR 1 \







# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "SSL_joint_upper2" \
# 								--multi 0 --singlev2 1 --joint_inf 1  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_specific/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '2' \
# 								--textAR 1 \






# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "SSL_joint_upper" \
# 								--multi 0 --singlev2 1 --joint_inf 1  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_specific/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '2' \
# 								--textAR 0 \









# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "SSL_joint_lower" \
# 								--multi 0 --singlev2 1 --joint_inf 1  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_specific/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '0' \
# 								--textAR 0 \









# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "textFFN_qydetach_ssl" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_specific/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 5000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '1' \
# 								--textAR 0 \






# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "textFFN_qydetach_upper" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_specific/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 5000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '2' \
# 								--textAR 0 \






# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "textFFN_qydetach" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_specific/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 5000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '0' \
# 								--textAR 0 \




# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "SSL_upper_2" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_specific/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 5000 --viz_steps 5000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '2' \




# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "SSL_lower" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 1 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_specific/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 5000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '0' \




# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "SSL_upper" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 1 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 0 \
# 								--which_gpu '0' \
# 								--quick_check 0 \
# 								--quick_check_data "$HOME/VL/two_objects_large/quick_stuff.pkl" \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_specific/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 5000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '2' \







# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "SSL_joint_blue" \
# 								--multi 1  --joint_inf 1  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--quick_check 0 \
# 								--quick_check_data "$HOME/VL/two_objects_large/quick_stuff.pkl" \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_specific/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 5000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \





# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "SSL_marg_blue" \
# 								--multi 1  --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--quick_check 0 \
# 								--quick_check_data "$HOME/VL/two_objects_large/quick_stuff.pkl" \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_specific/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 5000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \








# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "test" \
# 								--multi 1  --joint_inf 1  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--quick_check 0 \
# 								--quick_check_data "$HOME/VL/two_objects_large/quick_stuff.pkl" \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_specific/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 5000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \





# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "test" \
# 								--multi 1  --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--quick_check 0 \
# 								--quick_check_data "$HOME/VL/two_objects_large/quick_stuff.pkl" \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_specific/params/" \
# 								--model_load_step 0 \
# 								--display_step 5 --trainingplot_steps 500 --viz_steps 5000 \
# 								--start_storing_data_step 1 --save_params_step 50000 \





# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "SSL_joint" \
# 								--multi 1  --joint_inf 1  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--quick_check 0 \
# 								--quick_check_data "$HOME/VL/two_objects_large/quick_stuff.pkl" \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_specific/params/" \
# 								--model_load_step 0 \
# 								--display_step 1000 --trainingplot_steps 5000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \








# /home/ccremer/anaconda3/bin/python SSL.py --exp_name "test" \
# 								--multi 1  --joint_inf 1  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--quick_check 0 \
# 								--quick_check_data "$HOME/VL/two_objects_large/quick_stuff.pkl" \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_specific/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 500 --viz_steps 5000 \
# 								--start_storing_data_step 1 --save_params_step 50000 \




# /home/ccremer/anaconda3/bin/python CLEVR/SSL.py --exp_name "SSL_specific" \
# 								--multi 1  --joint_inf 0  --flow 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--quick_check 0 \
# 								--quick_check_data "$HOME/VL/two_objects_large/quick_stuff.pkl" \




# /home/ccremer/anaconda3/bin/python CLEVR/SSL.py --exp_name "test" \
# 								--multi 1  --joint_inf 0  --flow 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--quick_check 0 \
# 								--quick_check_data "$HOME/VL/two_objects_large/quick_stuff.pkl" \







# /home/ccremer/anaconda3/bin/python CLEVR/SSL.py --exp_name "SSL_detach_bluecyan_x5_reverse" \
# 								--multi 1  --joint_inf 0  --flow 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--quick_check 0 \
# 								--quick_check_data "$HOME/VL/two_objects_large/quick_stuff.pkl" \


