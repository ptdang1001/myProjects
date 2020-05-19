#!/bin/bash
module switch python/2.7.16 python/3.6.8
for baseAddNorm in {0..1}; do
  for minusMean in {0..1}; do
    for ((xn=20; xn<=40; xn+=5)); do
      for ((stdBias=0; stdBias<=10; stdBias+=2)); do
        for ((baseNumThreshold = 10; baseNumThreshold <= 200; baseNumThreshold +=20 )); do
          jobNum=$(squeue -u pdang | awk '$4=="pdang"' | wc -l)
          if [ $jobNum -le 1 ]; then
            sbatch sbatch.sh ${baseAddNorm} ${minusMean} ${xn} ${stdBias} ${baseNumThreshold}
          else
            while [ $jobNum -gt 1 ]; do
              sleep 30s
              jobNum=$(squeue -u pdang | awk '$4=="pdang"' | wc -l)
            done
            sbatch sbatch.sh ${baseAddNorm} ${minusMean} ${xn} ${stdBias} ${baseNumThreshold}
          fi
        done
      done
    done
  done
done

