#!/bin/bash

#SBATCH -J predL1
#SBATCH -p dl
#SBATCH -o log/overlap/res/testL1_%j.csv
#SBATCH -e log/overlap/err/testL1_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pdang@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --gres=gpu:p100:1
#SBATCH --time=00:10:00

#module load 
python testL1.py --xn $1 --noiseMBias $2 --noiseStdBias $3 --noiseNorm $4
