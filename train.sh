#!/bin/bash
#SBATCH --account=def-shirani
#SBATCH --mem=5G
#SBATCH --time=0-00:05
#SBATCH --mail-type=ALL
#SBATCH --mail-user=graham.slurm.hassanih@gmail.com
#SBATCH --output=job_slurm/%J.train.out
#SBATCH --job-name=train
echo "Starting run at: `date`"
./main.py\
  --epoch 1\
  --batch_size 128\
  --lr 0.001\
  --use_gpu 0\
  --phase train\
  --checkpoint_dir ./checkpoint\
  --sample_dir ./sample_results\
  --test_dir ./test_results

