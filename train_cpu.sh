#!/bin/bash
#SBATCH --account=def-shirani
#SBATCH --mem=2G

#SBATCH --time=0-00:40
#SBATCH --mail-type=ALL
#SBATCH --mail-user=graham.slurm.hassanih@gmail.com
#SBATCH --output=job_slurm/%J.train.out
#SBATCH --job-name=tr_cpu
echo "Starting run at: `date`"
model_name="msr_dual_net"
log_dir="./logs/"$model_name"-cpu"
source /home/hassanih/project/hassanih/ENV/bin/activate
tensorboard --logdir=$log_dir --port=8008 &
./main.py\
  --epoch 1000\
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
  --model_name $model_name\
  --hdf5_path ./data/mb2014_bin/data_da1_W512_H512_p64_s750_b16.hdf5


