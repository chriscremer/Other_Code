#! /bin/bash


#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=3GB
#SBATCH --job-name=myJob
#SBATCH --output=/h/ccremer/Documents/VAE2_exps/slurm_outputs/slurm_%j.out




# export LD_LIBRARY_PATH=/pkgs/cuda-9.2/lib64:$LD_LIBRARY_PATH
export PATH=$PATH:/h/ccremer/anaconda3/bin
# export PATH=/pkgs/anaconda3/bin:$PATH
source activate test_env

conda info --envs


python3 run_vae_cifar.py --exp_name "vae_test_gpu" \
								--z_size 384 --batch_size 64 \
								--enc_res_blocks 3 --dec_res_blocks 64 --n_prior_flows 25 \
								--which_gpu '0' \
								--data_dir "$HOME/Documents/" \
								--save_to_dir "$HOME/Documents/VAE2_exps/" \
								--display_step 500 --start_storing_data_step 2001 \
								--viz_steps 5000  --trainingplot_steps 5000 \
								--save_params_step 50000 --max_steps 400000 \
								--warmup_steps 20000 --continue_training 0 \
								--params_load_dir "$HOME/Documents/VAE2_exps/vae_test_prior_noperm/params/" \
								--model_load_step 0 \
								--save_output 1



source deactivate






# dfddafs
# srun --gres=gpu:1 -c 2 --mem=3G -p gpu -u --output=/h/ccremer/Documents/VAE2_exps/slurm_%j.out --job-name=myTest run_vws.sh

# conda info --envs

# #SBATCH --partition=cpu
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu
# #SBATCH --cpus-per-task=1
# #SBATCH --time=5:00:00
# #SBATCH --mem=4GB
# #SBATCH --job-name=myTest
# #SBATCH --output=slurm_%j.out

# # which python3
# stdbuf -i0 -o0 -e0 python3 run_vae_cifar.py --exp_name "vae_test_slurm3" \
# 								--z_size 384 --batch_size 64 \
# 								--enc_res_blocks 3 --dec_res_blocks 3  --n_prior_flows 5 \
# 								--which_gpu '0' \
# 								--data_dir "$HOME/Documents/" \
# 								--save_to_dir "$HOME/Documents/VAE2_exps/" \
# 								--display_step 500 --start_storing_data_step 2001 \
# 								--viz_steps 5000  --trainingplot_steps 5000 \
# 								--save_params_step 50000 --max_steps 100000 \
# 								--warmup_steps 20000 --continue_training 0 \

# to run: sbatch run_slurm.sh
