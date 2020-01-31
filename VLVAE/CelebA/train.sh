




/home/ccremer/anaconda3/bin/python train_CelebA.py --exp_name "vlvae_celebA_eval_attach_vs_detach" \
								--input_size 64 \
								--joint_inf 0  --flow_int 1 --batch_size 20 \
								--w_logpy 100 --w_logpx .1 --w_logqy 1 --max_beta 1 --z_size 50 \
								--data_dir "$HOME/VL/data/celebA/"  \
								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
								--just_classifier 0 \
								--train_classifier 0 \
								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/celeba_classifier/params" \
								--classifier_load_step 60000 \
								--which_gpu '1' \
								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
								--model_load_step 0 \
								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
								--start_storing_data_step 2001 --save_params_step 40000 \
								--textAR 1 \
								--max_steps 400000 \
								--qy_detach 0 \
								--seed 2 \
								--eval_attach_vs_detach 1








# /home/ccremer/anaconda3/bin/python train_qys2.py --exp_name "qy_celeba_eval" \
# 								--input_size 64 \
# 								--joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 100 --w_logpx .1 --w_logqy 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/celebA/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/celeba_classifier/params" \
# 								--classifier_load_step 60000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/vlvae_agg_detach_celebA/params/" \
# 								--model_load_step 400000 \
# 								--display_step 100 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 10000 \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'true' \
# 								--seed 2 \
# 								--eval_qy 1












# /home/ccremer/anaconda3/bin/python train_qys2.py --exp_name "qy_celeba_true_seed2" \
# 								--input_size 64 \
# 								--joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 100 --w_logpx .1 --w_logqy 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/celebA/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/celeba_classifier/params" \
# 								--classifier_load_step 60000 \
# 								--which_gpu '3' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/vlvae_agg_detach_celebA/params/" \
# 								--model_load_step 400000 \
# 								--display_step 100 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 10000 \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'true' \
# 								--seed 2



# /home/ccremer/anaconda3/bin/python train_qys2.py --exp_name "qy_celeba_agg_seed2" \
# 								--input_size 64 \
# 								--joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 100 --w_logpx .1 --w_logqy 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/celebA/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/celeba_classifier/params" \
# 								--classifier_load_step 60000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/vlvae_agg_detach_celebA/params/" \
# 								--model_load_step 400000 \
# 								--display_step 100 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 10000 \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'agg' \
# 								--seed 2




# /home/ccremer/anaconda3/bin/python train_qys2.py --exp_name "qy_celeba_agg_seed1" \
# 								--input_size 64 \
# 								--joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 100 --w_logpx .1 --w_logqy 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/celebA/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/celeba_classifier/params" \
# 								--classifier_load_step 60000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/vlvae_agg_detach_celebA/params/" \
# 								--model_load_step 400000 \
# 								--display_step 100 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 10000 \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'agg' \
# 								--seed 1






# /home/ccremer/anaconda3/bin/python train_qys2.py --exp_name "qy_celeba_true_seed1" \
# 								--input_size 64 \
# 								--joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 100 --w_logpx .1 --w_logqy 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/celebA/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/celeba_classifier/params" \
# 								--classifier_load_step 60000 \
# 								--which_gpu '2' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/vlvae_agg_detach_celebA/params/" \
# 								--model_load_step 400000 \
# 								--display_step 100 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 10000 \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'true' \
# 								--seed 1









# /home/ccremer/anaconda3/bin/python train_CelebA.py --exp_name "vlvae_celebA_attach_seed2" \
# 								--input_size 64 \
# 								--joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 100 --w_logpx .1 --w_logqy 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/celebA/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/celeba_classifier/params" \
# 								--classifier_load_step 60000 \
# 								--which_gpu '3' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 40000 \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 0 \
# 								--seed 2




# /home/ccremer/anaconda3/bin/python train_CelebA.py --exp_name "vlvae_celebA_attach_seed1" \
# 								--input_size 64 \
# 								--joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 100 --w_logpx .1 --w_logqy 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/celebA/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/celeba_classifier/params" \
# 								--classifier_load_step 60000 \
# 								--which_gpu '2' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 40000 \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 0 \
# 								--seed 1





# /home/ccremer/anaconda3/bin/python train_CelebA.py --exp_name "vlvae_celebA_detach_seed2" \
# 								--input_size 64 \
# 								--joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 100 --w_logpx .1 --w_logqy 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/celebA/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/celeba_classifier/params" \
# 								--classifier_load_step 60000 \
# 								--which_gpu '3' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 40000 \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--seed 2






# /home/ccremer/anaconda3/bin/python train_CelebA.py --exp_name "vlvae_celebA_detach_seed1" \
# 								--input_size 64 \
# 								--joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 100 --w_logpx .1 --w_logqy 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/celebA/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/celeba_classifier/params" \
# 								--classifier_load_step 60000 \
# 								--which_gpu '3' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 40000 \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--seed 1






# /home/ccremer/anaconda3/bin/python train_CelebA.py --exp_name "vlvae_celebA_testacc" \
# 								--input_size 64 \
# 								--joint_inf 0  --flow_int 1 --batch_size 32 \
# 								--w_logpy 1. --w_logpx 1. --w_logqy 1. --max_beta 1. --z_size 50 \
# 								--data_dir "$HOME/VL/data/celebA/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/celeba_classifier/params" \
# 								--classifier_load_step 60000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 40000 \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1






# /home/ccremer/anaconda3/bin/python train_CelebA.py --exp_name "vlvae_celebA_noweighting" \
# 								--input_size 64 \
# 								--joint_inf 0  --flow_int 1 --batch_size 32 \
# 								--w_logpy 1. --w_logpx 1. --w_logqy 1. --max_beta 1. --z_size 50 \
# 								--data_dir "$HOME/VL/data/celebA/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/celeba_classifier/params" \
# 								--classifier_load_step 60000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 40000 \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1








# /home/ccremer/anaconda3/bin/python train_qys2.py --exp_name "qy_celeba_true" \
# 								--input_size 64 \
# 								--joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 100 --w_logpx .1 --w_logqy 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/celebA/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/celeba_classifier/params" \
# 								--classifier_load_step 60000 \
# 								--which_gpu '3' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/vlvae_agg_detach_celebA/params/" \
# 								--model_load_step 400000 \
# 								--display_step 100 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 10000 \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'true'







# /home/ccremer/anaconda3/bin/python train_qys2.py --exp_name "qy_celeba_agg" \
# 								--input_size 64 \
# 								--joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 100 --w_logpx .1 --w_logqy 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/celebA/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/celeba_classifier/params" \
# 								--classifier_load_step 60000 \
# 								--which_gpu '2' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/vlvae_agg_detach_celebA/params/" \
# 								--model_load_step 400000 \
# 								--display_step 100 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 10000 \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1 \
# 								--qy_type 'agg'








# /home/ccremer/anaconda3/bin/python train_CelebA.py --exp_name "vlvae_agg_detach_celebA" \
# 								--input_size 64 \
# 								--joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 100 --w_logpx .1 --w_logqy 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/celebA/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/celeba_classifier/params" \
# 								--classifier_load_step 60000 \
# 								--which_gpu '0' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 40000 \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 1




# /home/ccremer/anaconda3/bin/python train_CelebA.py --exp_name "vlvae_agg_attach_celebA" \
# 								--input_size 64 \
# 								--joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 100 --w_logpx .1 --w_logqy 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/celebA/"  \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/celeba_classifier/params" \
# 								--classifier_load_step 60000 \
# 								--which_gpu '1' \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/fig1_2D_train/params/" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 2000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 40000 \
# 								--textAR 1 \
# 								--max_steps 400000 \
# 								--qy_detach 0








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






# /home/ccremer/anaconda3/bin/python train_qys.py --exp_name "test_models" \
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
# 								--max_steps 200000 \
# 								--qy_type 'neither' \
# 								--learn_prior 1











# /home/ccremer/anaconda3/bin/python train_qy_not_prior.py --exp_name "celeb_train_qy_agg_400k" \
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
# 								--display_step 1000 --trainingplot_steps 5000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/celeba_classifier/params" \
# 								--classifier_load_step 60000 \
# 								--max_steps 400000 \
# 								--qy_type 'agg' \
# 								--learn_prior 0





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
# 								--qy_type 'agg' \
# 								--learn_prior 0




# /home/ccremer/anaconda3/bin/python train_qys.py --exp_name "test_models" \
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
# 								--max_steps 200000 \
# 								--qy_type 'neither' \
# 								--learn_prior 1











# /home/ccremer/anaconda3/bin/python train_qys.py --exp_name "celeba_qys_learnprior_flow6_2" \
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
# 								--max_steps 200000 \
# 								--qy_type 'neither' \
# 								--learn_prior 1







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
# 								--max_steps 200000 \
# 								--qy_type 'true' \
# 								--learn_prior 0








# /home/ccremer/anaconda3/bin/python train_qy_not_prior.py --exp_name "celeba_qys_learn_true" \
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
# 								--max_steps 200000 \
# 								--qy_type 'true' \
# 								--learn_prior 0




# /home/ccremer/anaconda3/bin/python train_qy_not_prior.py --exp_name "celeba_qys_learn_agg" \
# 								--input_size 64 \
# 								--joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 1 --w_logpx 1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/celebA/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--which_gpu '0' \
# 								--quick_check 0 \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/celeba_3/params/" \
# 								--model_load_step 250000 \
# 								--display_step 500 --trainingplot_steps 5000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/celeba_classifier/params" \
# 								--classifier_load_step 60000 \
# 								--max_steps 200000 \
# 								--qy_type 'agg' \
# 								--learn_prior 0





# /home/ccremer/anaconda3/bin/python train_qys.py --exp_name "celeba_qys_learnprior_flow6" \
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
# 								--max_steps 200000 \
# 								--qy_type 'neither' \
# 								--learn_prior 1




# /home/ccremer/anaconda3/bin/python train_qys.py --exp_name "celeba_qys_learnprior_flow4" \
# 								--input_size 64 \
# 								--joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 100 --w_logpx .1 --max_beta 1 --z_size 50 \
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
# 								--max_steps 200000 \
# 								--qy_type 'neither' \
# 								--learn_prior 1




# /home/ccremer/anaconda3/bin/python train_qys.py --exp_name "celeba_qys_true" \
# 								--input_size 64 \
# 								--joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 100 --w_logpx .1 --max_beta 1 --z_size 50 \
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
# 								--max_steps 200000 \
# 								--qy_type 'true' \
# 								--learn_prior 0





# /home/ccremer/anaconda3/bin/python train_qys.py --exp_name "celeba_qys_agg" \
# 								--input_size 64 \
# 								--joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 100 --w_logpx .1 --max_beta 1 --z_size 50 \
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
# 								--max_steps 200000 \
# 								--qy_type 'agg' \
# 								--learn_prior 0



# /home/ccremer/anaconda3/bin/python train_CelebA.py --exp_name "celeba_4_withclassifier" \
# 								--input_size 64 \
# 								--joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 100 --w_logpx .1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/celebA/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--which_gpu '1' \
# 								--quick_check 0 \
# 								--params_load_dir "" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 5000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/celeba_classifier/params" \
# 								--classifier_load_step 60000 \
# 								--max_steps 200000








# /home/ccremer/anaconda3/bin/python train_CelebA.py --exp_name "celeb_viz_entropy" \
# 								--input_size 64 \
# 								--joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 100 --w_logpx .1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/celebA/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--which_gpu '1' \
# 								--quick_check 0 \
# 								--params_load_dir "$HOME/Documents/VLVAE_exps/celeba_3/params/" \
# 								--model_load_step 250000 \
# 								--display_step 1 --trainingplot_steps 5000 --viz_steps 1 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \
# 								--textAR 1









# /home/ccremer/anaconda3/bin/python train_CelebA.py --exp_name "celeba_3" \
# 								--input_size 64 \
# 								--joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 100 --w_logpx .1 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/celebA/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--which_gpu '0' \
# 								--quick_check 0 \
# 								--params_load_dir "" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 5000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \










# /home/ccremer/anaconda3/bin/python train_CelebA.py --exp_name "celeba_2" \
# 								--input_size 64 \
# 								--joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/celebA/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--which_gpu '0' \
# 								--quick_check 0 \
# 								--params_load_dir "" \
# 								--model_load_step 0 \
# 								--display_step 500 --trainingplot_steps 5000 --viz_steps 10000 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \











# /home/ccremer/anaconda3/bin/python train_CelebA.py --exp_name "test" \
# 								--input_size 64 \
# 								--joint_inf 0  --flow_int 1 --batch_size 20 \
# 								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
# 								--data_dir "$HOME/VL/data/celebA/" \
# 								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
# 								--just_classifier 0 \
# 								--train_classifier 0 \
# 								--which_gpu '1' \
# 								--quick_check 0 \
# 								--params_load_dir "" \
# 								--model_load_step 0 \
# 								--display_step 50 --trainingplot_steps 500 --viz_steps 500 \
# 								--start_storing_data_step 2001 --save_params_step 50000 \








