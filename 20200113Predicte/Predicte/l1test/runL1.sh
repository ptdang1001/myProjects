#!/bin/bash

for ((xn=10;xn<=16;xn=xn+3))
do
    for ((noiseMBias=0;noiseMBias<=50;noiseMBias=noiseMBias+10))
    do
        for ((noiseStdBias=0;noiseStdBias<=20;noiseStdBias=noiseStdBias+5))
        do  
            for ((noiseNorm=10;noiseNorm<=120;noiseNorm=noiseNorm+10))
            do  
                jobNum=$(squeue -u pdang | awk '$4=="pdang"' | wc -l)
                if [ $jobNum -le 1 ]
                then
                    sbatch sbatchL1.sh ${xn} ${noiseMBias} ${noiseStdBias} ${noiseNorm}
                    sleep 30s
                else
                    while [ $jobNum -gt 1 ]
                    do
                        sleep 30s
                        jobNum=$(squeue -u pdang | awk '$4=="pdang"' | wc -l)
                    done
                    sbatch sbatchL1.sh ${xn} ${noiseMBias} ${noiseStdBias} ${noiseNorm}
               fi
            done
        done
    done
done
