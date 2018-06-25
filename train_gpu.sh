#!/bin/bash
#SBATCH --account=def-shirani
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=0-01:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=graham.slurm.hassanih@gmail.com
#SBATCH --output=job_slurm/%J.train.out
#SBATCH --job-name=train_gpu
echo "Starting run at: `date`"
source /home/hassanih/project/hassanih/ENV/bin/activate
./main.py\
  --epoch 100\
  --batch_size 32\
  --lr 0.001\
  --use_gpu 1\
  --phase train\
  --checkpoint_dir ./checkpoint\
  --sample_dir ./sample_results\
  --test_dir ./test_results\
  --n 3\
  --v 1,10,100\
  --K 6\
  --fpxl ./data/mb2014_bin/pa_L_p64_b32_da1.npy\
  --fpxr ./data/mb2014_bin/pa_R_p64_b32_da1.npy\
  --fpyl ./data/mb2014_bin/gt_L_p64_b32_da1.npy\
  --eval_every_epoch 1

