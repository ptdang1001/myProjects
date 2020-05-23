#!/bin/bash
module switch python/2.7.16 python/3.6.8
for minusMean in {0..1}; do
  for ((xn = 20; xn <= 40; xn += 5)); do
    for ((stdBias = 0; stdBias <= 10; stdBias += 5)); do
      for ((numThreshold = 7; numThreshold <= 100; numThreshold += 10)); do
        jobNum=$(squeue -u pdang | awk '$2=="dl-debug" && $4=="pdang"' | wc -l)
        if [ $jobNum -le 1 ]; then
          sbatch sbatch.sh ${minusMean} ${xn} ${stdBias} ${numThreshold}
        else
          while [ $jobNum -gt 1 ]; do
            sleep 30s
            jobNum=$(squeue -u pdang | awk '$2=="dl-debug" && $4=="pdang"' | wc -l)
          done
          sbatch sbatch.sh ${minusMean} ${xn} ${stdBias} ${numThreshold}
        fi
      done
    done
  done
done
