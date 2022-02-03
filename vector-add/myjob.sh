#!/bin/bash
#SBATCH -t 5:00
#SBATCH -n 1 
#SBATCH --gres=gpu:1 
#SBATCH --reservation=jupyterhub_course_jhlsrf011_2022-01-31
#SBATCH --partition=gpu_shared_course
#SBATCH --mem=100000M

./vector-add 
./vector-add 256
./vector-add 1024
./vector-add 65536
./vector-add 1000000

./vector-add 655360 128
./vector-add 655360 256
./vector-add 655360 1024
./vector-add 655360 65536

./vector-add 655360 512 1
./vector-add 655360 512 2
./vector-add 655360 512 3