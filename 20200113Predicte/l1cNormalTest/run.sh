#!/bin/bash

for minusMean in ${0..1}
do
    for ((xn=7;xn<=16;xn=xn+3))
    do
        for ((normBias=0;normBias<=5;normBias=normBias+1))
        do  
            jobNum=$(squeue -u pdang | awk '$4=="pdang"' | wc -l)
            if [ $jobNum -le 1 ]
            then
                sbatch sbatchL1C.sh ${minusMean} ${xn} ${normBias}
                sleep 30s
            else
                while [ $jobNum -gt 1 ]
                do
                    sleep 30s
                    jobNum=$(squeue -u pdang | awk '$4=="pdang"' | wc -l)
                done
                sbatch sbatchL1C.sh ${minusMean} ${xn} ${normBias}
            fi
        done
    done
done
