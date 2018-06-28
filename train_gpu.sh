#!/bin/bash
#SBATCH --account=def-shirani
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=0-05:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=graham.slurm.hassanih@gmail.com
#SBATCH --output=job_slurm/%J.train.out
#SBATCH --job-name=train_gpu
echo "Starting run at: `date`"
source /home/hassanih/project/hassanih/ENV/bin/activate
./main.py\
  --epoch 300\
  --batch_size 64\
  --lr 0.001\
  --use_gpu 1\
  --phase train\
  --ckpt_dir ./checkpoint\
  --eval_dir ./eval_results\
  --test_dir ./test_results\
  --n 2\
  --v 1,10\
  --K 4\
  --eval_every_epoch 5\
  --model_name msr_net\
  --hdf5_path ./data/mb2014_bin/data_da1_W512_H512_p64_s128_b64.hdf5

