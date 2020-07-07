#!/bin/bash
for ((k= 50; k<= 500; k+= 50)); do
  jobNum=$(squeue -u pdang | awk '$2=="dl-debug" && $4=="pdang"' | wc -l)
  if [ $jobNum -le 1 ]; then
    sbatch sbatch.sh ${k}
  else
    while [ $jobNum -gt 1 ]; do
      sleep 30s
      jobNum=$(squeue -u pdang | awk '$2=="dl-debug" && $4=="pdang"' | wc -l)
    done
    sbatch sbatch.sh ${k}
  fi
done
