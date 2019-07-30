
#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12GB
#SBATCH --job-name=myJob
#SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
export PATH=$PATH:/h/ccremer/anaconda3/bin
source activate test_env
# conda info --envs
python3 train2.py  --exp_name "test_cifar_cov2" \
								--vws 0 \
								--which_gpu '0' \
								--dataset 'cifar' \
								--data_dir "$HOME/Documents/" \
								--save_to_dir "$HOME/Documents/glow_clevr/" \
								--batch_size 16 \
								--load_step 0 \
								--load_dir "$HOME/Documents/glow_clevr/clevr_confirmworks/params/" \
								--print_every 200 \
								--curveplot_every 2000 \
								--plotimages_every 2000 \
								--save_every 50000 \
								--max_steps 200000 \
								--n_levels 3 \
								--depth 16 \
								--hidden_channels 256 \
								--coupling 'additive' \
								--permutation 'shuffle' \
								--base_dist 'Gauss' \
								--lr 2e-4\
								--quick 1 \
								--save_output 0\





# #!/bin/sh
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=12GB
# #SBATCH --job-name=myJob
# #SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# # conda info --envs
# python3 train2.py  --exp_name "quick_test" \
# 								--vws 0 \
# 								--which_gpu '1' \
# 								--dataset 'cifar' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 16 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/clevr_confirmworks/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 1 \
# 								--save_every 50000 \
# 								--max_steps 200000 \
# 								--n_levels 3 \
# 								--depth 16 \
# 								--hidden_channels 256 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'Gauss' \
# 								--lr 2e-4\
# 								--quick 1 \
# 								--save_output 0\












# #!/bin/sh
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=12GB
# #SBATCH --job-name=myJob
# #SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# # conda info --envs
# python3 train2.py  --exp_name "cifar_flow" \
# 								--vws 0 \
# 								--which_gpu '0' \
# 								--dataset 'cifar' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 16 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/clevr_confirmworks/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 50000 \
# 								--max_steps 200000 \
# 								--n_levels 3 \
# 								--depth 16 \
# 								--hidden_channels 256 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'Gauss' \
# 								--lr 2e-4\
# 								--quick 1 \
# 								--save_output 0\




















# #!/bin/sh
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=12GB
# #SBATCH --job-name=myJob
# #SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# # conda info --envs
# python3 train2.py  --exp_name "clevr_affine_Gauss_fulldataset" \
# 								--vws 0 \
# 								--which_gpu '0' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 16 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/clevr_confirmworks/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 50000 \
# 								--max_steps 200000 \
# 								--n_levels 3 \
# 								--depth 16 \
# 								--hidden_channels 256 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'MoG' \
# 								--lr 2e-4\
# 								--quick 0 \
# 								--save_output 0\











# #!/bin/sh
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=12GB
# #SBATCH --job-name=myJob
# #SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# # conda info --envs
# python3 train2.py  --exp_name "clevr_affine_MoG_fulldataset" \
# 								--vws 0 \
# 								--which_gpu '1' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 16 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/clevr_confirmworks/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 50000 \
# 								--max_steps 200000 \
# 								--n_levels 3 \
# 								--depth 16 \
# 								--hidden_channels 256 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'MoG' \
# 								--lr 2e-4\
# 								--quick 0 \
# 								--save_output 0\












# #!/bin/sh
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=12GB
# #SBATCH --job-name=myJob
# #SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# # conda info --envs
# python3 train2.py  --exp_name "clevr_oneimage_test_MoG" \
# 								--vws 0 \
# 								--which_gpu '0' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 16 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/clevr_confirmworks/params/" \
# 								--print_every 10 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 50 \
# 								--save_every 50000 \
# 								--max_steps 200000 \
# 								--n_levels 3 \
# 								--depth 16 \
# 								--hidden_channels 256 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'MoG' \
# 								--lr 2e-4\
# 								--quick 1 \
# 								--save_output 0\
















# #!/bin/sh
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=12GB
# #SBATCH --job-name=myJob
# #SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# # conda info --envs
# python3 train2.py  --exp_name "clevr_oneimage_affine" \
# 								--vws 0 \
# 								--which_gpu '1' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 16 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/clevr_confirmworks/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 50000 \
# 								--max_steps 200000 \
# 								--n_levels 3 \
# 								--depth 16 \
# 								--hidden_channels 256 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--lr 2e-4\
# 								--quick 0 \
# 								--save_output 0\







# #!/bin/sh
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=12GB
# #SBATCH --job-name=myJob
# #SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# # conda info --envs
# python3 train2.py  --exp_name "clevr_oneimage" \
# 								--vws 0 \
# 								--which_gpu '1' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 16 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/clevr_confirmworks/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 50000 \
# 								--max_steps 200000 \
# 								--n_levels 3 \
# 								--depth 16 \
# 								--hidden_channels 256 \
# 								--coupling 'additive' \
# 								--permutation 'shuffle' \
# 								--lr 2e-4\
# 								--quick 0 \
# 								--save_output 0\









# #!/bin/sh
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=12GB
# #SBATCH --job-name=myJob
# #SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# # conda info --envs
# python3 train2.py  --exp_name "clevr_confirmworks" \
# 								--vws 0 \
# 								--which_gpu '0' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 16 \
# 								--load_step 180000 \
# 								--load_dir "$HOME/Documents/glow_clevr/clevr_confirmworks/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 40000 \
# 								--max_steps 200000 \
# 								--n_levels 3 \
# 								--depth 16 \
# 								--hidden_channels 256 \
# 								--coupling 'additive' \
# 								--permutation 'shuffle' \
# 								--lr 2e-5\
# 								--quick 0 \
# 								--save_output 0\













# #!/bin/sh
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=12GB
# #SBATCH --job-name=myJob
# #SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# # conda info --envs
# python3 train2.py  --exp_name "clevr_confirmworks_minmode" \
# 								--vws 0 \
# 								--which_gpu '0' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 16 \
# 								--load_step 180000 \
# 								--load_dir "$HOME/Documents/glow_clevr/clevr_confirmworks/params/" \
# 								--print_every 100 \
# 								--curveplot_every 1000 \
# 								--plotimages_every 100 \
# 								--save_every 40000 \
# 								--max_steps 200000 \
# 								--n_levels 3 \
# 								--depth 16 \
# 								--hidden_channels 256 \
# 								--coupling 'additive' \
# 								--permutation 'shuffle' \
# 								--lr 2e-5\
# 								--quick 0 \
# 								--save_output 0\



















# #!/bin/sh
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=12GB
# #SBATCH --job-name=myJob
# #SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# # conda info --envs
# python3 train2.py  --exp_name "clevr_withspline" \
# 								--vws 1 \
# 								--which_gpu '0' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 16 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/glow_sigmoid/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 40000 \
# 								--max_steps 200000 \
# 								--n_levels 3 \
# 								--depth 15 \
# 								--hidden_channels 256 \
# 								--coupling 'linear_spline' \
# 								--permutation 'shuffle' \
# 								--lr 2e-4\
# 								--quick 0 \
# 								--save_output 1\















# #!/bin/sh
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=12GB
# #SBATCH --job-name=myJob
# #SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# conda info --envs
# python3 train2.py  --exp_name "clevr_lessgaussclamping" \
# 								--vws 1 \
# 								--which_gpu '0' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 16 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/glow_sigmoid/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 40000 \
# 								--max_steps 200000 \
# 								--n_levels 3 \
# 								--depth 16 \
# 								--hidden_channels 256 \
# 								--coupling 'additive' \
# 								--permutation 'shuffle' \
# 								--lr 2e-4\
# 								--quick 0 \
# 								--save_output 1\







# #!/bin/sh
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=2GB
# #SBATCH --job-name=myJob
# #SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# conda info --envs
# python3 train2.py  --exp_name "clevr_test" \
# 								--vws 1 \
# 								--which_gpu '0' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 16 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/glow_sigmoid/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 40000 \
# 								--max_steps 200000 \
# 								--n_levels 3 \
# 								--depth 16 \
# 								--hidden_channels 256 \
# 								--coupling 'additive' \
# 								--permutation 'shuffle' \
# 								--lr 2e-4\
# 								--quick 1 \
# 								--save_output 1\




# python train2.py  --exp_name "clevr_confirmworks" \
# 								--vws 1 \
# 								--which_gpu '0' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 16 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/glow_sigmoid/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 30000 \
# 								--max_steps 200000 \
# 								--n_levels 3 \
# 								--depth 16 \
# 								--hidden_channels 256 \
# 								--coupling 'additive' \
# 								--permutation 'shuffle' \
# 								--lr 2e-4 \
# 								--save_output 0 \
# 								--quick 0 \








# python train2.py  --exp_name "clevr_smalldata_linearspline" \
# 								--vws 1 \
# 								--which_gpu '0' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 16 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/glow_sigmoid/params/" \
# 								--print_every 100 \
# 								--curveplot_every 1000 \
# 								--plotimages_every 1000 \
# 								--save_every 30000 \
# 								--max_steps 200000 \
# 								--n_levels 3 \
# 								--depth 16 \
# 								--hidden_channels 64 \
# 								--coupling 'linear_spline' \
# 								--permutation 'shuffle' \
# 								--lr 1e-4 \
# 								--save_output 0 \
# 								--quick 1 \










# python train2.py  --exp_name "clevr_smalldata_leaky" \
# 								--vws 1 \
# 								--which_gpu '0' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 16 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/glow_sigmoid/params/" \
# 								--print_every 100 \
# 								--curveplot_every 1000 \
# 								--plotimages_every 100 \
# 								--save_every 3000 \
# 								--max_steps 200000 \
# 								--n_levels 3 \
# 								--depth 16 \
# 								--hidden_channels 64 \
# 								--coupling 'additive' \
# 								--permutation 'shuffle' \
# 								--lr 1e-4 \
# 								--save_output 0 \
# 								--quick 1 \









# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=2GB
# #SBATCH --job-name=myJob
# #SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# conda info --envs
# python3 train2.py  --exp_name "clevr_test" \
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
# 								--depth 8 \
# 								--hidden_channels 64 \
# 								--coupling 'additive' \
# 								--permutation 'shuffle' \
# 								--lr 1e-4\
# 								--quick 1 \
# 								--save_output 1\





# python train2.py  --exp_name "clevr_betterflow2_filter7" \
# 								--vws 1 \
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
# 								--depth 16 \
# 								--hidden_channels 256 \
# 								--coupling 'additive' \
# 								--permutation 'shuffle' \
# 								--lr 1e-4\
# 								--quick 1 \




# python train2.py  --exp_name "clevr_betterflow2" \
# 								--vws 1 \
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
# 								--depth 16 \
# 								--hidden_channels 256 \
# 								--coupling 'additive' \
# 								--permutation 'shuffle' \
# 								--quick 0 \






# python train2.py  --exp_name "clevr_betterflow" \
# 								--which_gpu '0' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 32 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/glow_sigmoid/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 5000 \
# 								--save_every 40000 \
# 								--max_steps 200000 \
# 								--n_levels 3 \
# 								--depth 32 \
# 								--hidden_channels 64 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--quick 0 \







# python train2.py  --exp_name "clevr_128channels_32depth_continue_training" \
# 								--which_gpu '0' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 32 \
# 								--load_step 80000 \
# 								--load_dir "$HOME/Documents/glow_clevr/clevr_128channels_32depth/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 5000 \
# 								--save_every 40000 \
# 								--max_steps 200000 \
# 								--n_levels 3 \
# 								--depth 32 \
# 								--hidden_channels 128 \
# 								--coupling 'additive' \
# 								--permutation 'shuffle' \
# 								--quick 0 \






# python train2.py  --exp_name "clevr_spline" \
# 								--which_gpu '0' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 16 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/clevr_128channels_32depth/params/" \
# 								--print_every 100 \
# 								--curveplot_every 1000 \
# 								--plotimages_every 300 \
# 								--save_every 40000 \
# 								--max_steps 200000 \
# 								--n_levels 3 \
# 								--depth 8 \
# 								--hidden_channels 64 \
# 								--coupling 'spline' \
# 								--permutation 'shuffle' \
# 								--quick 1 \






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
# 								--depth 16 \
# 								--hidden_channels 64 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--quick 1 \




# python train2.py  --exp_name "clevr_spline" \
# 								--which_gpu '0' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 16 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/clevr_128channels_32depth/params/" \
# 								--print_every 100 \
# 								--curveplot_every 1000 \
# 								--plotimages_every 1000 \
# 								--save_every 40000 \
# 								--max_steps 200000 \
# 								--n_levels 3 \
# 								--depth 16 \
# 								--hidden_channels 128 \
# 								--coupling 'spline' \
# 								--permutation 'shuffle' \
# 								--quick 1 \








# python train2.py  --exp_name "clevr_128channels_32depth" \
# 								--which_gpu '0' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 32 \
# 								--load_step 80000 \
# 								--load_dir "$HOME/Documents/glow_clevr/clevr_128channels_32depth/params/" \
# 								--print_every 100 \
# 								--curveplot_every 1000 \
# 								--plotimages_every 1 \
# 								--save_every 40000 \
# 								--max_steps 200000 \
# 								--n_levels 3 \
# 								--depth 32 \
# 								--hidden_channels 128 \
# 								--coupling 'additive' \
# 								--permutation 'shuffle' \
# 								--quick 1 \








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





