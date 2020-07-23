#!/bin/bash
for ((k= 50; k<= 500; k+=100)); do
  jobNum=$(squeue -u pdang | awk '$2=="dl" && $4=="pdang"' | wc -l)
  if [ $jobNum -le 1 ]; then
    echo ${k}
    sbatch sbatch.sh ${k}
  else
    while [ $jobNum -gt 1 ]; do
      sleep 30s
      jobNum=$(squeue -u pdang | awk '$2=="dl" && $4=="pdang"' | wc -l)
    done
    echo ${k}
    sbatch sbatch.sh ${k}
  fi
done
