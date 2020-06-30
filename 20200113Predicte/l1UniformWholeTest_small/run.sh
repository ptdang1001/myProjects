#!/bin/bash
for ((xn = 50; xn <= 300; xn += 50)); do
  for ((baseTimes = 1; baseTimes <= 50; baseTimes += 5)); do
    for ((errorStdBias=0; errorStdBias<=9; errorStdBias+=3)); do
      jobNum=$(squeue -u pdang | awk '$2=="dl-debug" && $4=="pdang"' | wc -l)
      if [ $jobNum -le 1 ]; then
        sbatch sbatch.sh ${xn} ${baseTimes} ${errorStdBias}
      else
        while [ $jobNum -gt 1 ]; do
          sleep 30s
          jobNum=$(squeue -u pdang | awk '$2=="dl-debug" && $4=="pdang"' | wc -l)
        done
        sbatch sbatch.sh ${xn} ${baseTimes} ${errorStdBias}
      fi
    done
  done
done
