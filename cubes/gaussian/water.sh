#!/bin/bash

#SBATCH --job-name=water
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# 
#$ -S /bin/bash
source /opt/cluster/programs/g09_d.01/g09/bsd/g09.login.bash
#source /opt/Modules/etc/profile.modules

 
#set -xv
mkdir -p /scratch/$USER/water
export GAUSS_SCRDIR=/scratch/$USER/water

$GAUSSIAN_EXE /home/unibas/boittier/FF-Energy/cubes/test_jax/water.com /scratch/$USER/water/water.out

# don't delete the result file if not able to copy to fileserver 
cp /scratch/$USER/water/water.out /home/unibas/boittier/FF-Energy/cubes/test_jax/water.out 
status=$?
if [ $status -eq 0 ] 
then 
   rm -rf /scratch/$USER/water
else
   host=`/bin/hostname`
   /usr/bin/Mail -v -s "Error at end of batch job" $USER@verdi.cluster <<EOF

At the end of the batch job the system could not copy the output file
	$host:/scratch/$USER/water/water.out
to
	/home/unibas/boittier/FF-Energy/cubes/test_jax/water.out
Please copy this file by hand or inform the system manager.

EOF
 
fi
