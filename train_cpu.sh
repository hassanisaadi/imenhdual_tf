#!/bin/bash
#SBATCH --account=def-shirani
#SBATCH --mem=60G
#SBATCH --time=0-01:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=graham.slurm.hassanih@gmail.com
#SBATCH --output=job_slurm/%J.train.out
#SBATCH --job-name=train
echo "Starting run at: `date`"
./main.py\
  --epoch 2\
  --batch_size 32\
  --lr 0.001\
  --use_gpu 0\
  --phase train\
  --checkpoint_dir ./checkpoint\
  --sample_dir ./sample_results\
  --test_dir ./test_results\
  --n 2\
  --v 1,10\
  --K 2\
  --fpxl ./data/mb2014_bin/pa_L_p8_b16_da1.npy\
  --fpxr ./data/mb2014_bin/pa_R_p8_b16_da1.npy\
  --fpyl ./data/mb2014_bin/gt_L_p8_b16_da1.npy\
  --eval_every_epoch 1

