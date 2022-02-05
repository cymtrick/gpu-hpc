#!/bin/bash
#SBATCH -t 5:00
#SBATCH -n 1 
#SBATCH --gres=gpu:1 
#SBATCH --partition=gpu_shared_course
#SBATCH --mem=100000M

./vector-transform 
./vector-transform 256
./vector-transform 1024
./vector-transform 65536
./vector-transform 1000000
