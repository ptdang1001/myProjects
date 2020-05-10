#!/bin/bash

#SBATCH -J predL1C
#SBATCH -p dl
#SBATCH -o log/overlap/res/testL1c_%j.csv
#SBATCH -e log/overlap/err/testL1c_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pdang@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --gres=gpu:p100:1
#SBATCH --time=00:10:00

#module load 
python testL1C.py --minusMean $1 --xn $2 --normBias $3
