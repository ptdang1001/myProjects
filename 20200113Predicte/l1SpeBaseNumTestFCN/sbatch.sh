#!/bin/bash

#SBATCH -J L1Spe
#SBATCH -p dl
#SBATCH -o /N/project/zhangclab/pengtao/myProjectsDataRes/20200113Predicte/results/l1SpeBaseNumTestFCN/block1/res/testMain_%j.csv
#SBATCH -e /N/project/zhangclab/pengtao/myProjectsDataRes/20200113Predicte/results/l1SpeBaseNumTestFCN/block1/err/testMain_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pdang@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --gres=gpu:p100:1
#SBATCH --time=00:60:00

#module load
python testMain.py --minusMean $1 --xn $2 --stdBias $3 --numThreshold $4
