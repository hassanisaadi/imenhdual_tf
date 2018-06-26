#!/bin/bash
#SBATCH --account=def-shirani
#SBATCH --mem=50G
#SBATCH --time=0-00:15
#SBATCH --mail-type=ALL
#SBATCH --mail-user=graham.slurm.hassanih@gmail.com
#SBATCH --output=job_slurm/%J.gen_patches.out
module load python/2.7.14
module load gcc/5.4.0 opencv/2.4.13.3
source /home/hassanih/project/hassanih/ENV/bin/activate
./gen_patches.py\
  --src_dir ./data/mb2014_png/train\
  --dst_dir ./data/mb2014_bin\
  --data_aug_times 1\
  --patch_size 16\
  --step 0\
  --stride 1200\
  --batch_size 16
