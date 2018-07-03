#!/bin/bash
#SBATCH --account=def-shirani
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=0-12:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=graham.slurm.hassanih@gmail.com
#SBATCH --output=job_slurm/%J.train.out
#SBATCH --job-name=trgpu
echo "Starting run at: `date`"
model_name="msr_dual_net"
log_dir="./logs/"$model_name"-gpu"
source /home/hassanih/project/hassanih/ENV/bin/activate
tensorboard --logdir=$log_dir --port=8008 &
./main.py\
  --epoch 300\
  --batch_size 64\
  --lr 0.001\
  --use_gpu 1\
  --phase train\
  --ckpt_dir ./checkpoint\
  --eval_dir ./eval_results\
  --test_dir ./test_results\
  --n 3\
  --v 1,10,100\
  --K 4\
  --eval_every_epoch 10\
  --model_name $model_name\
  --hdf5_path ./data/mb2014_bin/data_da1_W512_H512_p64_s64_b64.hdf5


 # --hdf5_path ./data/mb2014_bin/data_da1_W512_H512_p64_s32_b64.hdf5
