###################################################################start dont change####
## carbonate
## qsub.sh
#!/bin/bash
#PBS -l nodes=1:ppn=16,mem=100gb,walltime=01:30:00
#PBS -l vmem=100gb
#PBS -M pdang@iu.edu
#PBS -m abe
# load required program
#source /N/u/pdang/Carbonate/.bashrc
python mmrf.py
