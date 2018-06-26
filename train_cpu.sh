#!/bin/bash
#SBATCH --account=def-shirani
#SBATCH --mem=100G
#SBATCH --time=0-03:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=graham.slurm.hassanih@gmail.com
#SBATCH --output=job_slurm/%J.train.out
#SBATCH --job-name=train
echo "Starting run at: `date`"
source /home/hassanih/project/hassanih/ENV/bin/activate
./main.py\
  --epoch 100\
  --batch_size 16\
  --lr 0.001\
  --use_gpu 0\
  --phase train\
  --checkpoint_dir ./checkpoint\
  --sample_dir ./sample_results\
  --test_dir ./test_results\
  --n 2\
  --v 1,10\
  --K 2\
  --fpxl ./data/mb2014_bin/pa_L_p16_b16_da1_s1200.npy\
  --fpxr ./data/mb2014_bin/pa_R_p16_b16_da1_s1200.npy\
  --fpyl ./data/mb2014_bin/gt_L_p16_b16_da1_s1200.npy\
  --eval_every_epoch 10\
  --model_name msr_net

