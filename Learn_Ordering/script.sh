





# VWS
export PATH=$PATH:/h/ccremer/anaconda3/bin
source activate test_env
python3 train9.py  --exp_name "learn_ordering_clevr_3" \
								--machine 'vws' \
								--which_gpu '0' \
								--dataset 'clevr' \
								--data_dir "$HOME/Documents/" \
								--save_to_dir "$HOME/Documents/glow_clevr/" \
								--batch_size 8 \
								--load_step 0 \
								--load_dir "$HOME/Documents/glow_clevr/learn_ordering_clevr_newversion_beta_2_alldata_continued/params/" \
								--print_every 200 \
								--curveplot_every 5000 \
								--plotimages_every 10000 \
								--save_every 10000 \
								--max_steps 200000 \
								--lr 1e-4 \
								--quick 0 \
								--save_output 0 \
								--sample 0 \
								--NLL_plot 0 \




# # VWS
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# python3 train9.py  --exp_name "learn_ordering_clevr_newversion_beta_2_alldata_continued_sample" \
# 								--machine 'vws' \
# 								--which_gpu '0' \
# 								--dataset 'clevr' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 8 \
# 								--load_step 170000 \
# 								--load_dir "$HOME/Documents/glow_clevr/learn_ordering_clevr_newversion_beta_2_alldata_continued/params/" \
# 								--print_every 200 \
# 								--curveplot_every 5000 \
# 								--plotimages_every 10000 \
# 								--save_every 10000 \
# 								--max_steps 200000 \
# 								--lr 1e-4 \
# 								--quick 1 \
# 								--save_output 0 \
# 								--sample 0 \
# 								--NLL_plot 1 \





# # VWS
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# python3 train8.py  --exp_name "learn_ordering_clevr_newversion_beta_2" \
# 								--machine 'vws' \
# 								--which_gpu '0' \
# 								--dataset 'clevr' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 8 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/learn_ordering_clevr_newversion_beta_2/params/" \
# 								--print_every 200 \
# 								--curveplot_every 5000 \
# 								--plotimages_every 5000 \
# 								--save_every 5000 \
# 								--max_steps 200000 \
# 								--lr 1e-4 \
# 								--quick 1 \
# 								--save_output 0 \
# 								# --dataset_size 2 \





# # VWS
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# python3 train8.py  --exp_name "learn_ordering_clevr_newversion_beta" \
# 								--machine 'vws' \
# 								--which_gpu '0' \
# 								--dataset 'clevr' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 8 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/FlowAR_clevr_test2/params/" \
# 								--print_every 400 \
# 								--curveplot_every 10000 \
# 								--plotimages_every 10000 \
# 								--save_every 25000 \
# 								--max_steps 200000 \
# 								--lr 1e-4 \
# 								--quick 1 \
# 								--save_output 0 \
# 								# --dataset_size 2 \




# # VWS
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# python3 train7.py  --exp_name "learn_ordering_clevr_newversion_2" \
# 								--machine 'vws' \
# 								--which_gpu '0' \
# 								--dataset 'clevr' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 8 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/FlowAR_clevr_test2/params/" \
# 								--print_every 200 \
# 								--curveplot_every 5000 \
# 								--plotimages_every 5000 \
# 								--save_every 25000 \
# 								--max_steps 200000 \
# 								--lr 1e-5 \
# 								--quick 1 \
# 								--save_output 0 \
# 								# --dataset_size 2 \


# # VWS
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# python3 train6.py  --exp_name "learn_ordering_clevr_first_5" \
# 								--machine 'vws' \
# 								--which_gpu '0' \
# 								--dataset 'clevr' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 8 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/FlowAR_clevr_test2/params/" \
# 								--print_every 200 \
# 								--curveplot_every 5000 \
# 								--plotimages_every 5000 \
# 								--save_every 200001 \
# 								--max_steps 200000 \
# 								--lr 1e-5 \
# 								--quick 1 \
# 								--save_output 0 \
# 								# --dataset_size 2 \




# # VWS
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# python3 train6.py  --exp_name "learn_ordering_clevr_first_4" \
# 								--machine 'vws' \
# 								--which_gpu '0' \
# 								--dataset 'clevr' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 8 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/FlowAR_clevr_test2/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 200001 \
# 								--max_steps 200000 \
# 								--lr 1e-5 \
# 								--quick 1 \
# 								--save_output 0 \
# 								# --dataset_size 2 \



# # VWS
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# python3 train6.py  --exp_name "learn_ordering_clevr_first_sampletopk" \
# 								--machine 'vws' \
# 								--which_gpu '0' \
# 								--dataset 'clevr' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 8 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/FlowAR_clevr_test2/params/" \
# 								--print_every 50 \
# 								--curveplot_every 500 \
# 								--plotimages_every 250 \
# 								--save_every 200001 \
# 								--max_steps 200000 \
# 								--lr 5e-5 \
# 								--quick 1 \
# 								--save_output 0 \
# 								# --dataset_size 2 \


# # VWS
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# python3 train5.py  --exp_name "flickr_oneimages_someflow" \
# 								--machine 'vws' \
# 								--which_gpu '0' \
# 								--dataset 'flickr' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 8 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/FlowAR_clevr_test2/params/" \
# 								--print_every 100 \
# 								--curveplot_every 20000 \
# 								--plotimages_every 400 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 12 \
# 								--hidden_channels 128 \
# 								--AR_resnets 5 \
# 								--AR_channels 32 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-3 \
# 								--quick 1 \
# 								--save_output 0 \
# 								--dataset_size 2 \

# # VWS
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# python3 train5.py  --exp_name "flickr_twoimages_someflow" \
# 								--machine 'vws' \
# 								--which_gpu '0' \
# 								--dataset 'flickr' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 8 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/FlowAR_clevr_test2/params/" \
# 								--print_every 100 \
# 								--curveplot_every 20000 \
# 								--plotimages_every 400 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 12 \
# 								--hidden_channels 128 \
# 								--AR_resnets 5 \
# 								--AR_channels 32 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-3 \
# 								--quick 1 \
# 								--save_output 0 \
# 								--dataset_size 2 \


# # VWS
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# python3 train5.py  --exp_name "diaggauss_flickr_fiftyimage" \
# 								--machine 'vws' \
# 								--which_gpu '0' \
# 								--dataset 'flickr' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 1 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/FlowAR_clevr_test2/params/" \
# 								--print_every 100 \
# 								--curveplot_every 20000 \
# 								--plotimages_every 1000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 12 \
# 								--hidden_channels 512 \
# 								--AR_resnets 5 \
# 								--AR_channels 32 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-3 \
# 								--quick 1 \
# 								--save_output 0 \
# 								--dataset_size 50 \





# # VWS
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# python3 train4.py  --exp_name "FlowAR_cifar_oneimage" \
# 								--machine 'vws' \
# 								--which_gpu '0' \
# 								--dataset 'cifar' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 1 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/FlowAR_clevr_test2/params/" \
# 								--print_every 100 \
# 								--curveplot_every 1000 \
# 								--plotimages_every 1000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 12 \
# 								--hidden_channels 512 \
# 								--AR_resnets 5 \
# 								--AR_channels 32 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-4 \
# 								--quick 1 \
# 								--save_output 0 \
# 								--dataset_size 1 \

# # VWS
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# python3 train4.py  --exp_name "FlowAR_flickr_oneimage_moresqueeze" \
# 								--machine 'vws' \
# 								--which_gpu '0' \
# 								--dataset 'flickr' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 1 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/FlowAR_clevr_test2/params/" \
# 								--print_every 100 \
# 								--curveplot_every 1000 \
# 								--plotimages_every 1000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 3 \
# 								--depth 12 \
# 								--hidden_channels 64 \
# 								--AR_resnets 5 \
# 								--AR_channels 32 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-4 \
# 								--quick 1 \
# 								--save_output 0 \




# # VWS
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# python3 train4.py  --exp_name "FlowAR_clevr_checkpreprocessing" \
# 								--machine 'vws' \
# 								--which_gpu '0' \
# 								--dataset 'clevr' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 8 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/FlowAR_clevr_test2/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 5000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 3 \
# 								--depth 12 \
# 								--hidden_channels 64 \
# 								--AR_resnets 5 \
# 								--AR_channels 32 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-4 \
# 								--quick 1 \
# 								--save_output 0\







# # VWS
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# python3 train4.py  --exp_name "FlowAR_flickr" \
# 								--machine 'vws' \
# 								--which_gpu '0' \
# 								--dataset 'flickr' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 2 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/FlowAR_clevr_test2/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 5000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 3 \
# 								--depth 11 \
# 								--hidden_channels 64 \
# 								--AR_resnets 5 \
# 								--AR_channels 32 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-4 \
# 								--quick 1 \
# 								--save_output 0\




# #!/bin/sh
# #VECTOR CLUSTER
# #SBATCH --gres=gpu:6
# #SBATCH --partition=gpu
# #SBATCH --cpus-per-task=12
# #SBATCH --mem=60GB
# #SBATCH --job-name=myJob
# #SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# python3 train4.py  --exp_name "FlowAR_clevr_checkpreprocessing" \
# 								--machine 'vector' \
# 								--which_gpu '0' \
# 								--dataset 'flickr' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 12 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/FlowAR_clevr_test2/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 5000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 3 \
# 								--depth 11 \
# 								--hidden_channels 64 \
# 								--AR_resnets 5 \
# 								--AR_channels 32 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-4\
# 								--quick 0 \
# 								--save_output 0\



# # VWS
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# python3 train4.py  --exp_name "FlowAR_clevr_checkpreprocessing" \
# 								--machine 'vws' \
# 								--which_gpu '0' \
# 								--dataset 'flickr' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 2 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/FlowAR_clevr_test2/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 5000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 3 \
# 								--depth 12 \
# 								--hidden_channels 64 \
# 								--AR_resnets 5 \
# 								--AR_channels 32 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-4 \
# 								--quick 1 \
# 								--save_output 0\







# #!/bin/sh
# #VECTOR CLUSTER
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=10GB
# #SBATCH --job-name=myJob
# #SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# python3 train3.py  --exp_name "FlowAR_clevr_logsdv3_fromstart" \
# 								--machine 'vector' \
# 								--which_gpu '0' \
# 								--dataset 'clevr' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 8 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/FlowAR_clevr_test2/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 32 \
# 								--hidden_channels 512 \
# 								--AR_resnets 5 \
# 								--AR_channels 128 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-4\
# 								--quick 0 \
# 								--save_output 0\






# #!/bin/sh
# #VECTOR CLUSTER
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=10GB
# #SBATCH --job-name=myJob
# #SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# python3 train3.py  --exp_name "FlowAR_clevr_test2_logsdv3" \
# 								--machine 'vector' \
# 								--which_gpu '0' \
# 								--dataset 'clevr' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 8 \
# 								--load_step 200000 \
# 								--load_dir "$HOME/Documents/glow_clevr/FlowAR_clevr_test2/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 32 \
# 								--hidden_channels 512 \
# 								--AR_resnets 5 \
# 								--AR_channels 128 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-5\
# 								--quick 0 \
# 								--save_output 0\




# # VWS
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# python3 train3.py  --exp_name "FlowAR_clevr_test2_trainwithnewlogsd" \
# 								--machine 'vws' \
# 								--which_gpu '0' \
# 								--dataset 'clevr' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 8 \
# 								--load_step 200000 \
# 								--load_dir "$HOME/Documents/glow_clevr/FlowAR_clevr_test2/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 32 \
# 								--hidden_channels 512 \
# 								--AR_resnets 5 \
# 								--AR_channels 128 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-5\
# 								--quick 0 \
# 								--save_output 0\




# # VWS
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# python3 train3.py  --exp_name "FlowAR_clevr_test2_sampling" \
# 								--machine 'vws' \
# 								--which_gpu '0' \
# 								--dataset 'clevr' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 8 \
# 								--load_step 200000 \
# 								--load_dir "$HOME/Documents/glow_clevr/FlowAR_clevr_test2/params/" \
# 								--print_every 10 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 1 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 32 \
# 								--hidden_channels 512 \
# 								--AR_resnets 5 \
# 								--AR_channels 128 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-5\
# 								--quick 1 \
# 								--save_output 0\
















# #!/bin/bash
# export PATH=$HOME/ccremer/anaconda3/bin:$PATH
# source activate test_env
# #VAUGHAN CLUSTER
# #SBATCH --gres=gpu:1
# #SBATCH --partition=p100
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=10GB
# #SBATCH --job-name=myJob
# #SBATCH --output=$HOME/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# python3 train3.py  --exp_name "FlowAR_clevr_cauchy" \
# 								--machine 'vaughn' \
# 								--which_gpu '0' \
# 								--dataset 'clevr' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 8 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/FlowAR_larger_larger/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 32 \
# 								--hidden_channels 512 \
# 								--AR_resnets 5 \
# 								--AR_channels 128 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-4 \
# 								--quick 0 \
# 								--save_output 0 \





# #!/bin/sh
# #VECTOR CLUSTER
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=10GB
# #SBATCH --job-name=myJob
# #SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# python3 train3.py  --exp_name "FlowAR_clevr_cauchy" \
# 								--machine 'vector' \
# 								--which_gpu '0' \
# 								--dataset 'clevr' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 8 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/FlowAR_larger_larger/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 32 \
# 								--hidden_channels 512 \
# 								--AR_resnets 5 \
# 								--AR_channels 128 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-4 \
# 								--quick 0 \
# 								--save_output 0 \







# #!/bin/sh
# #VECTOR CLUSTER
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=10GB
# #SBATCH --job-name=myJob
# #SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# python3 train3.py  --exp_name "FlowAR_clevr_test2" \
# 								--machine 'vector' \
# 								--which_gpu '0' \
# 								--dataset 'clevr' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 8 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/FlowAR_larger_larger/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 32 \
# 								--hidden_channels 512 \
# 								--AR_resnets 5 \
# 								--AR_channels 128 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-4 \
# 								--quick 0 \
# 								--save_output 0 \








# #!/bin/sh
# #VECTOR CLUSTER
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=10GB
# #SBATCH --job-name=myJob
# #SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# python3 train3.py  --exp_name "FlowAR_clevr_test_minsamples" \
# 								--machine 'vector' \
# 								--which_gpu '0' \
# 								--dataset 'clevr' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 16 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/FlowAR_larger_larger/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 32 \
# 								--hidden_channels 256 \
# 								--AR_resnets 5 \
# 								--AR_channels 64 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-4 \
# 								--quick 0 \
# 								--save_output 0 \







# # VWS
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# python3 train3.py  --exp_name "FlowAR_larger_larger_sample" \
# 								--vws 1 \
# 								--which_gpu '0' \
# 								--dataset 'cifar' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 32 \
# 								--load_step 200000 \
# 								--load_dir "$HOME/Documents/glow_clevr/FlowAR_larger_larger/params/" \
# 								--print_every 10 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 1 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 32 \
# 								--hidden_channels 1024 \
# 								--AR_resnets 5 \
# 								--AR_channels 128 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-5\
# 								--quick 0 \
# 								--save_output 0\













# # VWS
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# # conda info --envs
# python3 train3.py  --exp_name "FlowAR_clevr_testatt2" \
# 								--vws 1 \
# 								--which_gpu '0' \
# 								--dataset 'clevr' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 8 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/FlowAR_clevr_sdlowerlimit/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 32 \
# 								--hidden_channels 512 \
# 								--AR_resnets 3 \
# 								--AR_channels 64 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AttPrior' \
# 								--lr 1e-4\
# 								--quick 0 \
# 								--save_output 0\








# # VWS
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# # conda info --envs
# python3 train3.py  --exp_name "FlowAR_clevr_testatt" \
# 								--vws 1 \
# 								--which_gpu '0' \
# 								--dataset 'clevr' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 8 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/FlowAR_clevr_sdlowerlimit/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 32 \
# 								--hidden_channels 512 \
# 								--AR_resnets 3 \
# 								--AR_channels 64 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AttPrior' \
# 								--lr 1e-4\
# 								--quick 1 \
# 								--save_output 0\






# # VWS
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# # conda info --envs
# python3 train3.py  --exp_name "FlowAR_clevr_squeezedoutput" \
# 								--vws 1 \
# 								--which_gpu '0' \
# 								--dataset 'clevr' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 8 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/FlowAR_clevr_sdlowerlimit/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 32 \
# 								--hidden_channels 512 \
# 								--AR_resnets 3 \
# 								--AR_channels 64 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-4\
# 								--quick 1 \
# 								--save_output 0\










# # VWS
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# # conda info --envs
# python3 train3.py  --exp_name "FlowAR_clevr_sdlowerlimit" \
# 								--vws 1 \
# 								--which_gpu '0' \
# 								--dataset 'clevr' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 8 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/FlowAR_clevr/params/" \
# 								--print_every 400 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 4000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 32 \
# 								--hidden_channels 512 \
# 								--AR_resnets 5 \
# 								--AR_channels 64 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-4\
# 								--quick 0 \
# 								--save_output 0\






# # VWS
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# python3 train3.py  --exp_name "FlowAR_larger_larger_continued_quick" \
# 								--vws 1 \
# 								--which_gpu '0' \
# 								--dataset 'cifar' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 32 \
# 								--load_step 200000 \
# 								--load_dir "$HOME/Documents/glow_clevr/FlowAR_larger_larger/params/" \
# 								--print_every 1 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 1 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 32 \
# 								--hidden_channels 1024 \
# 								--AR_resnets 5 \
# 								--AR_channels 128 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-5\
# 								--quick 0 \
# 								--save_output 0\







# # VWS
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# # conda info --envs
# python3 train3.py  --exp_name "FlowAR_clevr_quickviz" \
# 								--vws 1 \
# 								--which_gpu '0' \
# 								--dataset 'clevr' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 8 \
# 								--load_step 200000 \
# 								--load_dir "$HOME/Documents/glow_clevr/FlowAR_clevr/params/" \
# 								--print_every 1 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 32 \
# 								--hidden_channels 512 \
# 								--AR_resnets 5 \
# 								--AR_channels 64 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 6e-10\
# 								--quick 1 \
# 								--save_output 0\








# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# # conda info --envs
# python3 train3.py  --exp_name "FlowAR_clevr" \
# 								--vws 1 \
# 								--which_gpu '0' \
# 								--dataset 'clevr' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 8 \
# 								--load_step 20000 \
# 								--load_dir "$HOME/Documents/glow_clevr/FlowAR_clevr/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 32 \
# 								--hidden_channels 512 \
# 								--AR_resnets 5 \
# 								--AR_channels 64 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 6e-5\
# 								--quick 0 \
# 								--save_output 0\





# #!/bin/sh
# #SBATCH --gres=gpu:2
# #SBATCH --partition=gpu
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=10GB
# #SBATCH --job-name=myJob
# #SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# # conda info --envs
# python3 train2.py  --exp_name "FlowAR_larger_larger_continued" \
# 								--vws 1 \
# 								--which_gpu '0' \
# 								--dataset 'cifar' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 32 \
# 								--load_step 200000 \
# 								--load_dir "$HOME/Documents/glow_clevr/FlowAR_larger_larger/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 32 \
# 								--hidden_channels 1024 \
# 								--AR_resnets 5 \
# 								--AR_channels 128 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-5\
# 								--quick 0 \
# 								--save_output 0\










# #!/bin/bash
# export PATH=$HOME/ccremer/anaconda3/bin:$PATH
# # echo $PATH
# source activate test_env
# # conda activate test_env
# # conda info --envs
# #VAUNGH
# #SBATCH --gres=gpu:4
# #SBATCH --partition=p100
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=12GB
# #SBATCH --job-name=myJob
# #SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# python3 train3.py  --exp_name "FlowAR_larger3x" \
# 								--vws 1 \
# 								--which_gpu 'all' \
# 								--dataset 'cifar' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 32 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/quick_test_AR_5/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 64 \
# 								--hidden_channels 1024 \
# 								--AR_resnets 5 \
# 								--AR_channels 256 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-4\
# 								--quick 0 \
# 								--save_output 1\









# # BOLTZ
# python3 train_logsumexp.py  --exp_name "test_LME_largerbatch_smalldataset" \
# 								--vws 0 \
# 								--which_gpu '1' \
# 								--dataset 'cifar' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 128 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/quick_test_AR_5/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 32 \
# 								--hidden_channels 256 \
# 								--AR_resnets 5 \
# 								--AR_channels 64 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-4\
# 								--quick 0 \
# 								--save_output 0\
# 								--dataset_size 200 \





# # BOLTZ
# python3 train_progressive.py  --exp_name "quick_test1" \
# 								--vws 0 \
# 								--which_gpu '0' \
# 								--dataset 'cifar' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 16 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/quick_test_AR_5/params/" \
# 								--print_every 1 \
# 								--curveplot_every -1 \
# 								--plotimages_every 20000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 32 \
# 								--hidden_channels 512 \
# 								--AR_resnets 5 \
# 								--AR_channels 90 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-4\
# 								--quick 0 \
# 								--save_output 0\










# #!/bin/sh
# #SBATCH --gres=gpu:2
# #SBATCH --partition=gpu
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=10GB
# #SBATCH --job-name=myJob
# #SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# conda activate test_env
# # conda info --envs
# python3 train2.py  --exp_name "FlowAR_larger_larger" \
# 								--vws 1 \
# 								--which_gpu '0' \
# 								--dataset 'cifar' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 32 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/quick_test_AR_5/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 32 \
# 								--hidden_channels 1024 \
# 								--AR_resnets 5 \
# 								--AR_channels 128 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-4\
# 								--quick 0 \
# 								--save_output 0\



# #!/bin/sh
# #SBATCH --gres=gpu:2
# #SBATCH --partition=gpu
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=10GB
# #SBATCH --job-name=myJob
# #SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# # conda info --envs
# python3 train2.py  --exp_name "FlowAR_larger_larger" \
# 								--vws 1 \
# 								--which_gpu '0' \
# 								--dataset 'cifar' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 32 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/quick_test_AR_5/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 32 \
# 								--hidden_channels 1024 \
# 								--AR_resnets 5 \
# 								--AR_channels 128 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-4\
# 								--quick 0 \
# 								--save_output 0\





# #!/bin/sh
# #SBATCH --gres=gpu:2
# #SBATCH --partition=gpu
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=10GB
# #SBATCH --job-name=myJob
# #SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# # conda info --envs
# python3 train2.py  --exp_name "FlowAR_larger_larger" \
# 								--vws 1 \
# 								--which_gpu 'all' \
# 								--dataset 'cifar' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 64 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/quick_test_AR_5/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 32 \
# 								--hidden_channels 1024 \
# 								--AR_resnets 5 \
# 								--AR_channels 128 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-4\
# 								--quick 0 \
# 								--save_output 1\







# #!/bin/sh
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu
# #SBATCH --cpus-per-task=1
# #SBATCH --mem=8GB
# #SBATCH --job-name=myJob
# #SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# # conda info --envs
# python3 train2.py  --exp_name "quick_test" \
# 								--vws 1 \
# 								--which_gpu '0' \
# 								--dataset 'cifar' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 32 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/quick_test_AR_5/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 32 \
# 								--hidden_channels 1024 \
# 								--AR_resnets 5 \
# 								--AR_channels 128 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-4\
# 								--quick 0 \
# 								--save_output 0\










# #!/bin/sh
# #SBATCH --gres=gpu:1
# #SBATCH --partition=p100
# #SBATCH --cpus-per-task=1
# #SBATCH --mem=8GB
# #SBATCH --job-name=myJob
# #SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# # export PATH=/h/ccremer/anaconda3/bin:$PATH
# export PATH=/scratch/ssd001/home/ccremer/anaconda3/bin:$PATH
# source activate test_env
# # conda info --envs
# python3 train2.py  --exp_name "flowAR_smallldataset" \
# 								--vws 1 \
# 								--which_gpu 'all' \
# 								--dataset 'cifar' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 32 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/quick_test_AR_5/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 32 \
# 								--hidden_channels 256 \
# 								--AR_resnets 5 \
# 								--AR_channels 64 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-4 \
# 								--quick 0 \
# 								--dataset_size 100 \
# 								--save_output 1 \




# #!/bin/sh
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu
# #SBATCH --cpus-per-task=1
# #SBATCH --mem=8GB
# #SBATCH --job-name=myJob
# #SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# # conda info --envs
# python3 train2.py  --exp_name "quick_test" \
# 								--vws 1 \
# 								--which_gpu '0' \
# 								--dataset 'cifar' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 32 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/quick_test_AR_5/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 1 \
# 								--hidden_channels 256 \
# 								--AR_resnets 5 \
# 								--AR_channels 128 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-4\
# 								--quick 0 \
# 								--save_output 0\
















# #!/bin/sh
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu
# #SBATCH --cpus-per-task=1
# #SBATCH --mem=8GB
# #SBATCH --job-name=myJob
# #SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# # conda info --envs
# python3 train2.py  --exp_name "flowAR_lessflow" \
# 								--vws 1 \
# 								--which_gpu '0' \
# 								--dataset 'cifar' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 32 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/quick_test_AR_5/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 1 \
# 								--hidden_channels 256 \
# 								--AR_resnets 5 \
# 								--AR_channels 128 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-4\
# 								--quick 0 \
# 								--save_output 0\







# #!/bin/sh
# #SBATCH --gres=gpu:2
# #SBATCH --partition=gpu
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=10GB
# #SBATCH --job-name=myJob
# #SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# # conda info --envs
# python3 train2.py  --exp_name "quick_test" \
# 								--vws 1 \
# 								--which_gpu '0' \
# 								--dataset 'cifar' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 100 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/quick_test_AR_5/params/" \
# 								--print_every 20 \
# 								--curveplot_every 100 \
# 								--plotimages_every 2000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 32 \
# 								--hidden_channels 128 \
# 								--AR_resnets 5 \
# 								--AR_channels 64 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 5e-5\
# 								--quick 0 \
# 								--save_output 0\













# #!/bin/sh
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu
# #SBATCH --cpus-per-task=1
# #SBATCH --mem=8GB
# #SBATCH --job-name=myJob
# #SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# # conda info --envs
# python3 train2.py  --exp_name "flowAR_larger" \
# 								--vws 1 \
# 								--which_gpu 'all' \
# 								--dataset 'cifar' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 32 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/quick_test_AR_5/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 32 \
# 								--hidden_channels 256 \
# 								--AR_resnets 5 \
# 								--AR_channels 128 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 1e-4\
# 								--quick 0 \
# 								--save_output 1\















# #!/bin/sh
# #SBATCH --gres=gpu:2
# #SBATCH --partition=gpu
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=10GB
# #SBATCH --job-name=myJob
# #SBATCH --output=/h/ccremer/Documents/glow_clevr/slurm_outputs/slurm_%j.out
# export PATH=$PATH:/h/ccremer/anaconda3/bin
# source activate test_env
# # conda info --envs
# python3 train2.py  --exp_name "continue_training_flowAR" \
# 								--vws 1 \
# 								--which_gpu 'all' \
# 								--dataset 'cifar' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 128 \
# 								--load_step 40000 \
# 								--load_dir "$HOME/Documents/glow_clevr/quick_test_AR_5/params/" \
# 								--print_every 200 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 20000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 32 \
# 								--hidden_channels 128 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 5e-5\
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
# # conda info --envs
# python3 train2.py  --exp_name "quick_test_AR" \
# 								--vws 0 \
# 								--which_gpu '0' \
# 								--dataset 'cifar' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/glow_clevr/" \
# 								--batch_size 32 \
# 								--load_step 0 \
# 								--load_dir "$HOME/Documents/glow_clevr/clevr_confirmworks/params/" \
# 								--print_every 20 \
# 								--curveplot_every 2000 \
# 								--plotimages_every 2000 \
# 								--save_every 50000 \
# 								--max_steps 200000 \
# 								--n_levels 1 \
# 								--depth 16 \
# 								--hidden_channels 128 \
# 								--coupling 'affine' \
# 								--permutation 'shuffle' \
# 								--base_dist 'AR' \
# 								--lr 2e-3\
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





