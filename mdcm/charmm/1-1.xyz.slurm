#!/bin/bash
#SBATCH --job-name=1-1.xyz
##SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1200

module load cmake/cmake-3.23.0-gcc-11.2.0-openmpi-4.1.3
charmm=/home/boittier/dev-release-dcm/build/cmake/charmm
$charmm < 1-1.xyz.inp > 1-1.xyz.inp.out