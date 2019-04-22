



/home/ccremer/anaconda3/bin/python train_classifier.py --exp_name "celeba_classifier" \
								--input_size 64 \
								--batch_size 20 \
								--data_dir "$HOME/VL/data/celebA/" \
								--save_to_dir "$HOME/Documents/VLVAE_exps/" \
								--which_gpu '1' \
								--display_step 50 --trainingplot_steps 5000 --viz_steps 10000 \
								--start_storing_data_step 2001 --save_params_step 10000 \
								--classifier_load_params_dir "$HOME/Documents/VLVAE_exps/celeba_classifier/params" \
								--classifier_load_step 50000 \











