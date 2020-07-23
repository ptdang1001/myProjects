#!/bin/bash
for ((k= 3; k<= 6; k+= 1)); do
  for n in 500 1000 2000; do
    jobNum=$(squeue -u pdang | awk '$2=="dl-debug" && $4=="pdang"' | wc -l)
    if [ $jobNum -le 1 ]; then
      sbatch debug_sbatch.sh ${k} ${n}
    else
      while [ $jobNum -gt 1 ]; do
        sleep 30s
        jobNum=$(squeue -u pdang | awk '$2=="dl-debug" && $4=="pdang"' | wc -l)
      done
      sbatch debug_sbatch.sh ${k} ${n}
    fi
  done 
done
