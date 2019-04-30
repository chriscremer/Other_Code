



/home/ccremer/anaconda3/bin/python ../interpolations/interpolations.py --exp_name "test_apr25_interpolations" \
								--multi 1 --singlev2 0 --joint_inf 0  --flow_int 1 --batch_size 20 \
								--w_logpy 200 --w_logpx .02 --max_beta 1 --z_size 50 \
								--data_dir "$HOME/VL/data/two_objects_no_occ/"  \
								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
								--just_classifier 0 \
								--train_classifier 0 \
								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/two_object_classifier_params/" \
								--classifier_load_step 150000 \
								--which_gpu '1' \
								--params_load_dir "$HOME/Documents/VLVAE_exps/test_22apr2019/params/" \
								--model_load_step 400000 \
								--display_step 500 --trainingplot_steps 4000 --viz_steps 10000 \
								--start_storing_data_step 2001 --save_params_step 30000 \
								--textAR 1 \
								--max_steps 400000 \
								--qy_detach 1 
								# --quick_check 1 \
								# --quick_check_data "$HOME/VL/two_objects_large/quick_stuff.pkl" 









