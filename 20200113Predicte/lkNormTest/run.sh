#!/bin/bash
for replace in {0..1}
do
    for minusMean in {0..1}
    do
        for ((xn=7;xn<=16;xn=xn+1))
        do
            for ((normBias=0;normBias<=5;normBias=normBias+1))
            do  
                jobNum=$(squeue -u pdang | awk '$4=="pdang"' | wc -l)
                if [ $jobNum -le 1 ]
                then
                    sbatch sbatch.sh ${replace} ${minusMean} ${xn} ${normBias}
                else
                    while [ $jobNum -gt 1 ]
                    do
                        sleep 30s
                        jobNum=$(squeue -u pdang | awk '$4=="pdang"' | wc -l)
                    done
                    sbatch sbatch.sh ${replace} ${minusMean} ${xn} ${normBias}
                fi
            done
        done
    done
done