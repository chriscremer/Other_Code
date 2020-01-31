




/home/ccremer/anaconda3/bin/python ../training_code/train_qys.py --exp_name "qy_clevr_eval" \
								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
								--just_classifier 0 \
								--train_classifier 0 \
								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
								--classifier_load_step 150000 \
								--which_gpu '1' \
								--params_load_dir "$HOME/Documents/VLVAE_exps/vlvae_agg_detach_qy1/params/" \
								--model_load_step 400000 \
								--display_step 100 --trainingplot_steps 1000 --viz_steps 10000 \
								--start_storing_data_step 2001 --save_params_step 20000 \
								--ssl_type '0' \
								--textAR 1 \
								--max_steps 400000 \
								--qy_detach 1 \
								--qy_type 'agg' \
								--learn_prior 0 \
								--seed 2 \
								--eval_qy 1





# /home/ccremer/anaconda3/bin/python ../training_code/train_jointVAE.py --exp_name "vlvae_clevr_eval_attach_detach" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --w_logqy 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/vlvae_agg_detach_qy1/params/" \
# 								--model_load_step 400000 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 40000 \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--seed 2 \
# 								--eval_attach_vs_detach 1








# /home/ccremer/anaconda3/bin/python ../training_code/train_qys.py --exp_name "qy_clevr_agg_seed2" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '3' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/vlvae_agg_detach_qy1/params/" \
# 								--model_load_step 400000 \
# 								--display_step 100 --trainingplot_steps 1000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 20000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'agg' \
# 								--learn_prior 0 \
# 								--seed 2



# /home/ccremer/anaconda3/bin/python ../training_code/train_qys.py --exp_name "qy_clevr_true_seed2" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/vlvae_agg_detach_qy1/params/" \
# 								--model_load_step 400000 \
# 								--display_step 100 --trainingplot_steps 1000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 20000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'true' \
# 								--learn_prior 0 \
# 								--seed 2




# /home/ccremer/anaconda3/bin/python ../training_code/train_qys.py --exp_name "qy_clevr_true_seed1" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/vlvae_agg_detach_qy1/params/" \
# 								--model_load_step 400000 \
# 								--display_step 100 --trainingplot_steps 1000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 20000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'true' \
# 								--learn_prior 0 \
# 								--seed 1





# /home/ccremer/anaconda3/bin/python ../training_code/train_qys.py --exp_name "qy_clevr_agg_seed1" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '2' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/vlvae_agg_detach_qy1/params/" \
# 								--model_load_step 400000 \
# 								--display_step 100 --trainingplot_steps 1000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 20000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'agg' \
# 								--learn_prior 0 \
# 								--seed 1








# /home/ccremer/anaconda3/bin/python ../training_code/train_jointVAE.py --exp_name "vlvae_clevr_detach_seed2" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --w_logqy 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 40000 \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--seed 2






# /home/ccremer/anaconda3/bin/python ../training_code/train_jointVAE.py --exp_name "vlvae_clevr_attach_seed2" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --w_logqy 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 40000 \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 0 \
# 								--seed 2



# /home/ccremer/anaconda3/bin/python ../training_code/train_jointVAE.py --exp_name "vlvae_clevr_attach_seed1" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --w_logqy 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 40000 \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 0 \
# 								--seed 1


# /home/ccremer/anaconda3/bin/python ../training_code/train_jointVAE.py --exp_name "vlvae_clevr_detach_seed1" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --w_logqy 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '2' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 40000 \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--seed 1









# /home/ccremer/anaconda3/bin/python ../training_code/train_jointVAE.py --exp_name "vlvae_clevr_weightanneal" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --w_logqy 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '2' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 40000 \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 







# /home/ccremer/anaconda3/bin/python ../training_code/train_jointVAE.py --exp_name "vlvae_clevr_noweighting_noisyimage" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 1 --w_logpx 1 --w_logqy 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '2' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 40000 \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1






# /home/ccremer/anaconda3/bin/python ../training_code/train_jointVAE.py --exp_name "vlvae_clevr_noweighting" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 1 --w_logpx 1 --w_logqy 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '3' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 40000 \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1









# /home/ccremer/anaconda3/bin/python ../training_code/train_qys.py --exp_name "qy_agg_new2" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/vlvae_agg_detach_qy1/params/" \
# 								--model_load_step 400000 \
# 								--display_step 100 --trainingplot_steps 1000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 20000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'agg' \
# 								--learn_prior 0




# /home/ccremer/anaconda3/bin/python ../training_code/train_qys.py --exp_name "qy_true_new2" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/vlvae_agg_detach_qy1/params/" \
# 								--model_load_step 400000 \
# 								--display_step 100 --trainingplot_steps 1000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 20000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'true' \
# 								--learn_prior 0


							

# /home/ccremer/anaconda3/bin/python ../training_code/train_qys.py --exp_name "qy_true_2flows_BN" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/vlvae_agg_detach_qy1/params/" \
# 								--model_load_step 400000 \
# 								--display_step 100 --trainingplot_steps 1000 --viz_steps 4000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'true' \
# 								--learn_prior 0




# /home/ccremer/anaconda3/bin/python ../training_code/train_qys.py --exp_name "qy_agg_2flows_BN" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/vlvae_agg_detach_qy1/params/" \
# 								--model_load_step 400000 \
# 								--display_step 100 --trainingplot_steps 1000 --viz_steps 4000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'agg' \
# 								--learn_prior 0







# /home/ccremer/anaconda3/bin/python ../training_code/train_qys.py --exp_name "qy_true_6flows_moreplots" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '2' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/vlvae_agg_detach_qy1/params/" \
# 								--model_load_step 400000 \
# 								--display_step 100 --trainingplot_steps 1000 --viz_steps 4000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'true' \
# 								--learn_prior 0



# /home/ccremer/anaconda3/bin/python ../training_code/train_qys.py --exp_name "qy_agg_6flows_moreplots" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/vlvae_agg_detach_qy1/params/" \
# 								--model_load_step 400000 \
# 								--display_step 100 --trainingplot_steps 1000 --viz_steps 4000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'agg' \
# 								--learn_prior 0







# /home/ccremer/anaconda3/bin/python ../training_code/train_qys.py --exp_name "qy_agg_3flows_moreplots" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/vlvae_agg_detach_qy1/params/" \
# 								--model_load_step 400000 \
# 								--display_step 100 --trainingplot_steps 1000 --viz_steps 4000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'agg' \
# 								--learn_prior 0




# /home/ccremer/anaconda3/bin/python ../training_code/train_qys.py --exp_name "qy_true_3flows_moreplots" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '3' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/vlvae_agg_detach_qy1/params/" \
# 								--model_load_step 400000 \
# 								--display_step 100 --trainingplot_steps 1000 --viz_steps 4000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'true' \
# 								--learn_prior 0




# /home/ccremer/anaconda3/bin/python ../training_code/train_qys.py --exp_name "qy_true_3flows" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '2' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/vlvae_agg_detach_qy1/params/" \
# 								--model_load_step 400000 \
# 								--display_step 100 --trainingplot_steps 1000 --viz_steps 4000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'true' \
# 								--learn_prior 0



# /home/ccremer/anaconda3/bin/python ../training_code/train_qys.py --exp_name "qy_agg_3flows" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/vlvae_agg_detach_qy1/params/" \
# 								--model_load_step 400000 \
# 								--display_step 100 --trainingplot_steps 1000 --viz_steps 4000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'agg' \
# 								--learn_prior 0



# /home/ccremer/anaconda3/bin/python ../training_code/train_qys.py --exp_name "qy_agg_6condflows" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/vlvae_agg_detach_qy1/params/" \
# 								--model_load_step 400000 \
# 								--display_step 100 --trainingplot_steps 1000 --viz_steps 4000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'agg' \
# 								--learn_prior 0




# /home/ccremer/anaconda3/bin/python ../training_code/train_qys.py --exp_name "qy_true_2_flows2" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/vlvae_agg_detach_qy1/params/" \
# 								--model_load_step 400000 \
# 								--display_step 100 --trainingplot_steps 1000 --viz_steps 4000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'true' \
# 								--learn_prior 0






# /home/ccremer/anaconda3/bin/python ../training_code/train_qys.py --exp_name "qy_true_2" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/vlvae_agg_detach_qy1/params/" \
# 								--model_load_step 400000 \
# 								--display_step 100 --trainingplot_steps 1000 --viz_steps 999999999 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'true' \
# 								--learn_prior 0












# /home/ccremer/anaconda3/bin/python ../training_code/train_qys.py --exp_name "test_models_agg" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '3' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/vlvae_agg_detach_qy1/params/" \
# 								--model_load_step 400000 \
# 								--display_step 500 --trainingplot_steps 4000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'agg' \
# 								--learn_prior 0












# /home/ccremer/anaconda3/bin/python ../training_code/train_jointVAE.py --exp_name "vlvae_agg_detach_qy1" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --w_logqy 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '3' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 40000 \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1





# /home/ccremer/anaconda3/bin/python ../training_code/train_jointVAE.py --exp_name "vlvae_agg_attach_qy1" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --w_logqy 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '2' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 40000 \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 0






# /home/ccremer/anaconda3/bin/python ../training_code/train_jointVAE.py --exp_name "vlvae_agg_attach" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --w_logqy 20 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 40000 \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 0





# /home/ccremer/anaconda3/bin/python ../training_code/train_jointVAE.py --exp_name "vlvae_agg_detach" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --w_logqy 20 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 40000 \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1




# /home/ccremer/anaconda3/bin/python ../training_code/train_jointVAE.py --exp_name "vlvae_qy20_2" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --w_logqy 20 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 4000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 30000 \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1



# /home/ccremer/anaconda3/bin/python fig1_2D.py --exp_name "fig1_2D_test1" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 1 --w_logpx 1 --max_beta 1 --z_size 2 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 0 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/train_2D_adaptiveB_3/params/" \
# 								--model_load_step 60000 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 300000






# /home/ccremer/anaconda3/bin/python train_prior.py --exp_name "test_models" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 1 --w_logpx 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/qy_detach/params/" \
# 								--model_load_step 400000 \
# 								--display_step 1000 --trainingplot_steps 4000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'neither' \
# 								--learn_prior 1






# /home/ccremer/anaconda3/bin/python train_qys.py --exp_name "test_models" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 1 --w_logpx 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/qy_detach/params/" \
# 								--model_load_step 400000 \
# 								--display_step 500 --trainingplot_steps 4000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'agg' \
# 								--learn_prior 0











# /home/ccremer/anaconda3/bin/python train_prior.py --exp_name "train_learn_prior_2" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 1 --w_logpx 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/qy_detach/params/" \
# 								--model_load_step 400000 \
# 								--display_step 1000 --trainingplot_steps 4000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'neither' \
# 								--learn_prior 1








# /home/ccremer/anaconda3/bin/python train_prior.py --exp_name "train_learn_prior_2" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 1 --w_logpx 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/qy_detach/params/" \
# 								--model_load_step 400000 \
# 								--display_step 1000 --trainingplot_steps 4000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'neither' \
# 								--learn_prior 1







# /home/ccremer/anaconda3/bin/python train_qys.py --exp_name "test_models" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 1 --w_logpx 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/qy_detach/params/" \
# 								--model_load_step 400000 \
# 								--display_step 500 --trainingplot_steps 4000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'true' \
# 								--learn_prior 0




# /home/ccremer/anaconda3/bin/python train_qys.py --exp_name "train_learn_qy_true_betterqy" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 1 --w_logpx 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/qy_detach/params/" \
# 								--model_load_step 400000 \
# 								--display_step 500 --trainingplot_steps 4000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'true' \
# 								--learn_prior 0






# /home/ccremer/anaconda3/bin/python train_qys.py --exp_name "train_learn_qy_agg_betterqy" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 1 --w_logpx 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/qy_detach/params/" \
# 								--model_load_step 400000 \
# 								--display_step 500 --trainingplot_steps 4000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'agg' \
# 								--learn_prior 0







# /home/ccremer/anaconda3/bin/python train_prior.py --exp_name "train_learn_prior_y" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 1 --w_logpx 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/qy_detach/params/" \
# 								--model_load_step 400000 \
# 								--display_step 500 --trainingplot_steps 4000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'neither' \
# 								--learn_prior 1






# /home/ccremer/anaconda3/bin/python train_prior.py --exp_name "train_learn_prior" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 1 --w_logpx 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/qy_detach/params/" \
# 								--model_load_step 400000 \
# 								--display_step 500 --trainingplot_steps 4000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'neither' \
# 								--learn_prior 1




# /home/ccremer/anaconda3/bin/python train_qys.py --exp_name "train_learn_qy_agg_1s" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 1 --w_logpx 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/qy_detach/params/" \
# 								--model_load_step 400000 \
# 								--display_step 500 --trainingplot_steps 4000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'agg' \
# 								--learn_prior 0





# /home/ccremer/anaconda3/bin/python train_qys.py --exp_name "train_learn_qy_true_1s" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 1 --w_logpx 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/qy_detach/params/" \
# 								--model_load_step 400000 \
# 								--display_step 500 --trainingplot_steps 4000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'true' \
# 								--learn_prior 0







# /home/ccremer/anaconda3/bin/python train_qys.py --exp_name "train_learn_prior_multiobject_flow6" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/qy_detach/params/" \
# 								--model_load_step 400000 \
# 								--display_step 500 --trainingplot_steps 4000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'neither' \
# 								--learn_prior 1











# /home/ccremer/anaconda3/bin/python fig1_2D.py --exp_name "fig1_2D_test1" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 2 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 0 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/train_2D_adaptiveB_3/params/" \
# 								--model_load_step 60000 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 300000 \







# /home/ccremer/anaconda3/bin/python train_qys.py --exp_name "train_learn_prior" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 100 --w_logpx .01 --max_beta 40 --z_size 2 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir  "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/"  \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/train_2D_adaptiveB_3/params/" \
# 								--model_load_step 60000 \
# 								--display_step 500 --trainingplot_steps 4000 --viz_steps 5000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'neither' \
# 								--learn_prior 1







# /home/ccremer/anaconda3/bin/python train_qys.py --exp_name "train_qys_2D_agg_withent" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 100 --w_logpx .01 --max_beta 40 --z_size 2 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir  "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/"  \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/train_2D_adaptiveB_3/params/" \
# 								--model_load_step 60000 \
# 								--display_step 500 --trainingplot_steps 4000 --viz_steps 5000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'agg'







# /home/ccremer/anaconda3/bin/python train_qys.py --exp_name "train_qys_2D_true_trainmode" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 2 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir  "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/"  \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/train_2D_adaptiveB_3/params/" \
# 								--model_load_step 60000 \
# 								--display_step 500 --trainingplot_steps 4000 --viz_steps 5000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'true'



# /home/ccremer/anaconda3/bin/python train_jointVAE.py --exp_name "train_2D_fixedB_dettached" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 100 --w_logpx .01 --max_beta 1 --z_size 2 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 4000 --viz_steps 5000 \
# 								--start_storing_data_step 2001 --save_params_step 30000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 60000 \
# 								--qy_detach 1







# /home/ccremer/anaconda3/bin/python train_qys.py --exp_name "train_qys_true5_singleobj" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 400 --w_logpx .04 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir  "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/"  \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_oneobject_marg_lower/params/" \
# 								--model_load_step 250000 \
# 								--display_step 500 --trainingplot_steps 4000 --viz_steps 5000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1





# /home/ccremer/anaconda3/bin/python train_qys.py --exp_name "train_qys_agg5_singleobj" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 400 --w_logpx .04 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir  "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/"  \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_oneobject_marg_lower/params/" \
# 								--model_load_step 250000 \
# 								--display_step 500 --trainingplot_steps 4000 --viz_steps 5000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1
















# /home/ccremer/anaconda3/bin/python train_qys.py --exp_name "train_qys_true4_difmodel" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/qy_detach/params/" \
# 								--model_load_step 400000 \
# 								--display_step 500 --trainingplot_steps 4000 --viz_steps 5000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1




# /home/ccremer/anaconda3/bin/python train_qys.py --exp_name "train_qys_true4_moresamples" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SUL_marg_twoobjects_lower/params/" \
# 								--model_load_step 100000 \
# 								--display_step 500 --trainingplot_steps 4000 --viz_steps 5000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1



# /home/ccremer/anaconda3/bin/python train_qys.py --exp_name "train_qys_agg4_moresamples" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SUL_marg_twoobjects_lower/params/" \
# 								--model_load_step 100000 \
# 								--display_step 500 --trainingplot_steps 4000 --viz_steps 5000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1





# /home/ccremer/anaconda3/bin/python train_qys.py --exp_name "train_qys_true3_initflow" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SUL_marg_twoobjects_lower/params/" \
# 								--model_load_step 100000 \
# 								--display_step 500 --trainingplot_steps 4000 --viz_steps 5000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1



 




# /home/ccremer/anaconda3/bin/python train_qys.py --exp_name "train_qys_agg3_initflow" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SUL_marg_twoobjects_lower/params/" \
# 								--model_load_step 100000 \
# 								--display_step 500 --trainingplot_steps 4000 --viz_steps 5000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1






 



# /home/ccremer/anaconda3/bin/python train_qys.py --exp_name "train_qys_agg2" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SUL_marg_twoobjects_lower/params/" \
# 								--model_load_step 100000 \
# 								--display_step 500 --trainingplot_steps 4000 --viz_steps 5000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1





# /home/ccremer/anaconda3/bin/python train_qys.py --exp_name "train_qys_agg" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/qy_detach/params/" \
# 								--model_load_step 400000 \
# 								--display_step 500 --trainingplot_steps 4000 --viz_steps 5000 \
# 								--start_storing_data_step 2001 --save_params_step 999999999 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1






# /home/ccremer/anaconda3/bin/python train_jointVAE.py --exp_name "train_2D_fixedB_dettached" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 100 --w_logpx .01 --max_beta 1 --z_size 2 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 4000 --viz_steps 5000 \
# 								--start_storing_data_step 2001 --save_params_step 30000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 60000 \
# 								--qy_detach 1



# /home/ccremer/anaconda3/bin/python train_jointVAE.py --exp_name "train_2D_fixedB_attached" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 100 --w_logpx .01 --max_beta 1 --z_size 2 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 4000 --viz_steps 5000 \
# 								--start_storing_data_step 2001 --save_params_step 30000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 60000 \
# 								--qy_detach 0







# /home/ccremer/anaconda3/bin/python train_jointVAE.py --exp_name "train_2D_adaptiveB_attached" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 2 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 4000 --viz_steps 5000 \
# 								--start_storing_data_step 2001 --save_params_step 30000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 60000 \
# 								--qy_detach 0





# /home/ccremer/anaconda3/bin/python train_jointVAE.py --exp_name "train_2D_adaptiveB_dettached" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 2 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 4000 --viz_steps 5000 \
# 								--start_storing_data_step 2001 --save_params_step 30000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 60000 \
# 								--qy_detach 1






# /home/ccremer/anaconda3/bin/python fig1_2D.py --exp_name "fig1_2D_test" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 2 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 0 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/train_2D_adaptiveB_3/params/" \
# 								--model_load_step 60000 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 300000 \




# /home/ccremer/anaconda3/bin/python fig1_2D.py --exp_name "fig1_2D_test" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 2 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 0 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/train_2D_1/params/" \
# 								--model_load_step 50000 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 300000 \







# /home/ccremer/anaconda3/bin/python train_jointVAE.py --exp_name "train_2D_adaptiveB_4" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta .001 --z_size 2 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 4000 --viz_steps 5000 \
# 								--start_storing_data_step 2001 --save_params_step 30000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 60000 \




# # longer warmup and lower beta
# /home/ccremer/anaconda3/bin/python train_VLAAE.py --exp_name "train_2D_VLAAE_3" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 100 --z_size 2 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 5000 \
# 								--start_storing_data_step 501 --save_params_step 20000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 60000 \







# /home/ccremer/anaconda3/bin/python train_jointVAE.py --exp_name "train_2D_adaptiveB_3" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 2 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 4000 --viz_steps 5000 \
# 								--start_storing_data_step 2001 --save_params_step 30000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 60000 \





# /home/ccremer/anaconda3/bin/python train_2DVLVAE.py --exp_name "train_2D_newidea" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 2 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 1000 --trainingplot_steps 2000 --viz_steps 5000 \
# 								--start_storing_data_step 2001 --save_params_step 30000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 60000 \





# /home/ccremer/anaconda3/bin/python train_VLAAE.py --exp_name "train_2D_VLAAE_longerwarmup" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1000 --z_size 2 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 5000 --viz_steps 5000 \
# 								--start_storing_data_step 501 --save_params_step 20000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 60000 \




# /home/ccremer/anaconda3/bin/python train_VLAAE.py --exp_name "train_2D_VLAAE" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1000 --z_size 2 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 5000 --viz_steps 5000 \
# 								--start_storing_data_step 501 --save_params_step 20000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 60000 \







# /home/ccremer/anaconda3/bin/python train_jointVAE.py --exp_name "all_1s" \
# 								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 1 --w_logpx 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/two_objects_no_occ/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 1000 --trainingplot_steps 2000 --viz_steps 15000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \



# /home/ccremer/anaconda3/bin/python fig1_2D.py --exp_name "fig1_2D_viz_2" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 2 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/train_2D_1/params/" \
# 								--model_load_step 50000 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 300000 \








# /home/ccremer/anaconda3/bin/python train_jointVAE.py --exp_name "train_2D_1" \
# 								--multi 0 --singlev2 1 --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 2 \
# 								--data_dir "$HOME/VL/data/clevr_single_object/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/PARAMS/single_object_classifier/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--ssl_type '0' \
# 								--textAR 1 \
# 								--max_steps 400000 \



# /home/ccremer/anaconda3/bin/python train_jointVAE.py --exp_name "test" \
# 								--multi 1  --joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 1000 --w_logpx .01 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_specific/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 5000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 --max_steps 400000 \
# 								--textAR 1  --qy_detach 1 \














# /home/ccremer/anaconda3/bin/python train_jointVAE.py --exp_name "test_pre_send" \
# 								--multi 1  --joint_inf 1  --flow_int 1 --batch_size 20 \
# 								--w_logpy 1000 --w_logpx .01 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/SSL_specific/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 5000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--textAR 1  --qy_detach 1 \









# /home/ccremer/anaconda3/bin/python train_jointVAE.py --exp_name "clevr_2" \
# 								--multi 1  --joint_inf 0  --flow_int 1 --batch_size 20 \
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









# /home/ccremer/anaconda3/bin/python CLEVR/train_jointVAE.py --exp_name "qy_KL" \
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








# /home/ccremer/anaconda3/bin/python CLEVR/train_jointVAE.py --exp_name "joint_no_occ_3" \
# 								--multi 1  --joint_inf 1  --flow 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--quick_check 0 \
# 								--quick_check_data "$HOME/VL/two_objects_large/quick_stuff.pkl" \












# /home/ccremer/anaconda3/bin/python CLEVR/train_jointVAE.py --exp_name "marg_no_occ_2" \
# 								--multi 1  --joint_inf 0  --flow 1 --batch_size 20 \
# 								--w_logpy 100 --w_logpx .01 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--quick_check 0 \
# 								--quick_check_data "$HOME/VL/two_objects_large/quick_stuff.pkl" \










# #to debug logpy
# /home/ccremer/anaconda3/bin/python CLEVR/train_jointVAE.py --exp_name "joint_inf_debug" \
# 								--multi 1  --joint_inf 1  --flow 1 --batch_size 20 \
# 								--w_logpy 1000 --w_logpx .1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/two_objects_large/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--quick_check 1 \
# 								--quick_check_data "$HOME/VL/two_objects_large/quick_stuff.pkl" \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/joint_inf/params/" \
# 								--model_load_step 150000 \







# /home/ccremer/anaconda3/bin/python CLEVR/train_jointVAE.py --exp_name "joint_inf_2" \
# 								--multi 1  --joint_inf 1  --flow 1 --batch_size 20 \
# 								--w_logpy 500 --w_logpx .05 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/two_objects_large/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '1' \
# 								--quick_check 0 \
# 								--quick_check_data "$HOME/VL/two_objects_large/quick_stuff.pkl" \


# /home/ccremer/anaconda3/bin/python CLEVR/train_jointVAE.py --exp_name "marg_inf_2" \
# 								--multi 1  --joint_inf 0  --flow 1 --batch_size 20 \
# 								--w_logpy 500 --w_logpx .05 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/two_objects_large/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--quick_check 0 \
# 								--quick_check_data "$HOME/VL/two_objects_large/quick_stuff.pkl" \


# /home/ccremer/anaconda3/bin/python CLEVR/train_jointVAE.py --exp_name "marg_inf_2" \
# 								--multi 1  --joint_inf 0  --flow 1 --batch_size 20 \
# 								--w_logpy 500 --w_logpx .05 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/two_objects_large/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
# 								--classifier_load_step 150000 \
# 								--which_gpu '0' \
# 								--quick_check 0 \
# 								--quick_check_data "$HOME/VL/two_objects_large/