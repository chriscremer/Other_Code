




python train2.py  --exp_name "clevr_128channels_32depth" \
								--which_gpu '0' \
								--data_dir "$HOME/Documents/" \
								--save_to_dir "$HOME/Documents/glow_clevr/" \
								--batch_size 32 \
								--load_step 80000 \
								--load_dir "$HOME/Documents/glow_clevr/clevr_128channels_32depth/params/" \
								--print_every 100 \
								--curveplot_every 1000 \
								--plotimages_every 1 \
								--save_every 40000 \
								--max_steps 200000 \
								--n_levels 3 \
								--depth 32 \
								--hidden_channels 128 \
								--coupling 'additive' \
								--permutation 'shuffle' \
								--quick 0 \




# python train2.py  --exp_name "clevr_betterflow2" \
# 								--vws 1 \
# 								--which_gpu '0' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 16 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/glow_sigmoid/params/" \
# 								--print_every 100 \
# 								--curveplot_every 1000 \
# 								--plotimages_every 2000 \
# 								--save_every 40000 \
# 								--max_steps 200000 \
# 								--n_levels 3 \
# 								--depth 64 \
# 								--hidden_channels 64 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--quick 0 \


# python train2.py  --exp_name "clevr_betterflow" \
# 								--which_gpu '0' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 32 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/glow_sigmoid/params/" \
# 								--print_every 100 \
# 								--curveplot_every 1000 \
# 								--plotimages_every 2000 \
# 								--save_every 40000 \
# 								--max_steps 200000 \
# 								--n_levels 3 \
# 								--depth 32 \
# 								--hidden_channels 64 \
# 								--quick 0 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \





# python train2.py  --exp_name "clevr_betterflow" \
# 								--which_gpu '0' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 16 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/glow_sigmoid/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 5000 \
# 								--save_every 40000 \
# 								--max_steps 200000 \
# 								--n_levels 3 \
# 								--depth 32 \
# 								--quick 0 \
# 								--coupling 'additive' \
# 								--permutation 'conv' \








# python train2.py  --exp_name "clevr_128channels_32depth" \
# 								--which_gpu '0' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 32 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/glow_sigmoid/params/" \
# 								--print_every 100 \
# 								--curveplot_every 1000 \
# 								--plotimages_every 2000 \
# 								--save_every 40000 \
# 								--max_steps 200000 \
# 								--n_levels 3 \
# 								--depth 32 \






# python train2.py  --exp_name "clevr_gettingprobofsamples" \
# 								--which_gpu '0' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 32 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/glow_sigmoid/params/" \
# 								--print_every 100 \
# 								--curveplot_every 1000 \
# 								--plotimages_every 2000 \
# 								--save_every 40000 \
# 								--max_steps 200000 \
# 								--n_levels 3 \
# 								--depth 32 \





