#! /bin/bash



# srun --gres=gpu:1 -c 2 --mem=3G -p gpu -u --output=/h/ccremer/Documents/VAE2_exps/slurm_%j.out --job-name=myTest run_vws.sh



#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=3GB
#SBATCH --job-name=myTest
#SBATCH --output=/h/ccremer/Documents/VAE2_exps/slurm_%j.out
#SBATCH --unbuffered

# export LD_LIBRARY_PATH=/pkgs/cuda-9.2/lib64:$LD_LIBRARY_PATH
export PATH=$PATH:/h/ccremer/anaconda3/bin
# export PATH=/pkgs/anaconda3/bin:$PATH
source activate test_env

conda info --envs

# which python3
stdbuf -i0 -o0 -e0 python3 run_vae_cifar.py --exp_name "vae_test_slurm3" \
								--z_size 384 --batch_size 64 \
								--enc_res_blocks 3 --dec_res_blocks 3  --n_prior_flows 5 \
								--which_gpu '0' \
								--data_dir "$HOME/Documents/" \
								--save_to_dir "$HOME/Documents/VAE2_exps/" \
								--display_step 500 --start_storing_data_step 2001 \
								--viz_steps 5000  --trainingplot_steps 5000 \
								--save_params_step 50000 --max_steps 100000 \
								--warmup_steps 20000 --continue_training 0 \

source deactivate

conda info --envs

#SBATCH --partition=cpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=myTest
#SBATCH --output=slurm_%j.out



