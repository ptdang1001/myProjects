#!/bin/bash

#SBATCH -J mmrf 
#SBATCH -p dl-debug 
#SBATCH -o  /N/project/zhangclab/pengtao/myProjectsDataRes/20200619Classification/results/res/mmrf_%j.csv
#SBATCH -e  /N/project/zhangclab/pengtao/myProjectsDataRes/20200619Classification/results/err/mmrf_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pdang@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --gres=gpu:p100:1
#SBATCH --time=00:55:00

#module load
python testMain.py #--minusMean $1 --xn $2 --stdBias $3 --numThreshold $4
