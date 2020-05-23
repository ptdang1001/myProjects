#!/bin/bash
module switch python/2.7.16 python/3.6.8

for minusMean in {0..1}
do
    for normBias in {0..1}
    do
      for ((baseNumThreshold=20;baseNumThreshold<=300;baseNumThreshold+=10))
      do
        jobNum=$(squeue -u pdang | awk '$4=="pdang"' | wc -l)
        if [ $jobNum -le 1 ]
        then
          sbatch sbatch.sh ${minusMean} ${normBias} ${baseNumThreshold}
        else
          while [ $jobNum -gt 1 ]
          do
            sleep 30s
            jobNum=$(squeue -u pdang | awk '$4=="pdang"' | wc -l)
          done
          sbatch sbatch.sh ${minusMean} ${normBias} ${baseNumThreshold}
        fi
      done
    done
done
