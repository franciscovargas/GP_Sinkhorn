#!/bin/bash
#SBATCH --time=11:59:00
#SBATCH --nodes=1


cd 
module load CCEnv arch/avx512 nixpkgs/16.09
module load python/3.7.0
source sch_bridge/bin/activate
cd $SCRATCH/GP_Sinkhorn/script/


counter=$(<sbatch/counter.txt)
dir_log="${SCRATCH}/GP_Sinkhorn/assets/result_dump/${counter}"
mkdir ${dir_log}
counter=$((counter+1))
echo $counter > sbatch/counter.txt


python EB_Dataset.py --sigma 2.5 --decay-sigma 0.95 --iteration 20 --log-dir $dir_log --sparse 0 --gp-prior 1
