#!/bin/bash
#SBATCH --account=def-shirani
#SBATCH --mem=20G
#SBATCH --time=0-00:45
#SBATCH --mail-type=ALL
#SBATCH --mail-user=graham.slurm.hassanih@gmail.com
#SBATCH --output=job_slurm/%J.train.out
#SBATCH --job-name=train
echo "Starting run at: `date`"
source /home/hassanih/project/hassanih/ENV/bin/activate
./main.py\
  --epoch 10\
  --batch_size 64\
  --lr 0.001\
  --use_gpu 0\
  --phase train\
  --ckpt_dir ./checkpoint\
  --eval_dir ./eval_results\
  --test_dir ./test_results\
  --n 2\
  --v 1,10\
  --K 2\
  --eval_every_epoch 2\
  --model_name simple_net\
  --hdf5_path ./data/mb2014_bin/data_da1_W512_H512_p64_s128_b64.hdf5
  #--hdf5_path ./data/mb2014_bin/data_da1_W300_H300_p16_s750_b8.hdf5
#  --hdf5_path ./data/mb2014_bin/data_da2_W2800_H1900_p64_s32_b32.hdf5


