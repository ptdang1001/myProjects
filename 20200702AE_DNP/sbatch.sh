#!/bin/bash

#SBATCH -J L1Spe
#SBATCH -p dl
#SBATCH -o /N/project/zhangclab/pengtao/myProjectsDataRes/20200702AE_DNP/results/res/main_%j.csv
#SBATCH -e /N/project/zhangclab/pengtao/myProjectsDataRes/20200702AE_DNP/results/err/main_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pdang@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --gres=gpu:p100:1
#SBATCH --time=24:00:00

#module load
python main.py #--k $1
