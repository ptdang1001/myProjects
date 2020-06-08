#!/bin/bash
module switch python/2.7.16 python/3.6.8
for ((xn = 50; xn <= 550; xn += 100)); do
  for ((stdBias = 0; stdBias <= 4; stdBias += 2)); do
    for ((baseTimes = 3; baseTimes <= 9; baseTimes += 3)); do
      jobNum=$(squeue -u pdang | awk '$2=="dl-debug" && $4=="pdang"' | wc -l)
      if [ $jobNum -le 1 ]; then
        sbatch sbatch.sh ${xn} ${stdBias} ${baseTimes}
      else
        while [ $jobNum -gt 1 ]; do
          sleep 30s
          jobNum=$(squeue -u pdang | awk '$2=="dl-debug" && $4=="pdang"' | wc -l)
        done
        sbatch sbatch.sh ${xn} ${stdBias} ${baseTimes}
      fi
    done
  done
done
