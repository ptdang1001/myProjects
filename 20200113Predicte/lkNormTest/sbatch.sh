#!/bin/bash

#SBATCH -J pred
#SBATCH -p dl
#SBATCH -o /N/slate/pdang/myProjectsDataRes/20200113Predicte/results/lkNormTest/noverlap/res/testMain_%j.csv
#SBATCH -e /N/slate/pdang/myProjectsDataRes/20200113Predicte/results/lkNormTest/noverlap/err/testMain_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pdang@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --gres=gpu:p100:1
#SBATCH --time=00:10:00

#module load
module switch python/2.7.16 python/3.6.8
#run testMain
python testMain.py --replace $1 --minusMean $2 --xn $3 --normBias $4
