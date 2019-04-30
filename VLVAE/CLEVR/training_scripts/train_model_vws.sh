



python3 train_jointVAE.py --exp_name "test_23apr2019_debuggingplotting" \
								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
								--data_dir "$HOME/vl_data/two_objects_large/" \
								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
								--just_classifier 0 \
								--train_classifier 0 \
								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/gaus_qy/params/" \
								--classifier_load_step 150000 \
								--which_gpu '0' \
								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_test/params/" \
								--model_load_step 0 \
								--display_step 500 --trainingplot_steps 4000 --viz_steps 10000 \
								--start_storing_data_step 2001 --save_params_step 50000 \
								--textAR 1 \
								--max_steps 400000 \








# python3 SSL.py --exp_name "SSL_oneobject_marg_test" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/vl_data/one_object_large/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/vl_data/one_object_large/single_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_specific/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \




# python3 train_jointVAE.py --exp_name "fig1_2D_2" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 2 \
# 								--data_dir "$HOME/vl_data/one_object_large/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/vl_data/one_object_large/single_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 300000 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 300000 \












# python3 train_jointVAE.py --exp_name "fig1_2D_train" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 2 \
# 								--data_dir "$HOME/vl_data/one_object_large/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/vl_data/one_object_large/single_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_test/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 300000 \







# python3 fig1_2D.py --exp_name "fig1_2D" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/vl_data/one_object_large/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/vl_data/one_object_large/single_object_classifier_params/" \
# 								--classifier_load_step 0 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 10000 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 5000 \
# 								--start_storing_data_step 2001 --save_params_step 5000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 5000 \
# 								# --quick_check 1 \
# 								# --quick_check_data "$HOME/vl_data/quick_stuff.pkl" 


# python3 train_jointVAE.py --exp_name "fig1_2D_train" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/vl_data/one_object_large/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/vl_data/one_object_large/single_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_test/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 10000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 10001 \








# python3 train_jointVAE.py --exp_name "fig1_2D_test" \
# 								--multi 0 --singlev2 1 --joint_inf 1  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/vl_data/one_object_large/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/vl_data/one_object_large/single_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_test/params/" \
# 								--model_load_step 10 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 20000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \







# python3 train_jointVAE.py --multi 1 --max_beta 1 --train_classifier 0 --w_logpy 1000 --w_logpx .1 --joint_inf 0 --flow 1  --exp_name flow_qy --quick_check 1

# python3 train_jointVAE.py --multi 1 --max_beta 1 --train_classifier 0 --w_logpy 1000 --w_logpx .1 --z_size 20 --joint_inf 0 --flow 1  --exp_name flow_qy


# python3 train_jointVAE.py --multi 1 --max_beta 1 --train_classifier 0 --w_logpy 1000 --w_logpx .1 --z_size 50 --joint_inf 0 --flow 1  --exp_name flow_qy_sum

# python3 CLEVR/train_jointVAE.py --multi 1 --max_beta 1 --train_classifier 1 --w_logpy 1000 --w_logpx .1 --z_size 50 --joint_inf 0  --exp_name test_new_format --quick_check 1


# python3 CLEVR/train_jointVAE.py --multi 1 --max_beta 1 --train_classifier 0 --model_load_step 150000 --w_logpy 1000 --w_logpx .1 --z_size 50 --joint_inf 0  --exp_name test_new_format --quick_check 1

# python3 CLEVR/train_jointVAE.py --exp_name test_new_format \
# 								--multi 1  --joint_inf 0  --flow 1\
# 								--w_logpy 1000 --w_logpx .1 --max_beta 1 --z_size 50 \
# 								--model_load_step 50000  --train_classifier 0 \
# 								--quick_check 1 


# python3 CLEVR/train_jointVAE.py --exp_name "test_new_format" \
# 								--multi 1  --joint_inf 0  --flow 1 --batch_size 20 \
# 								--w_logpy 1000 --w_logpx .1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/vl_data/two_objects_large/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/gaus_qy/params/" \
# 								--classifier_load_step 150000 \
# 								--quick_check 1 \









