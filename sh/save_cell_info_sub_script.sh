#!/bin/bash
#
# request resources:
#PBS -l nodes=1:ppn=16,walltime=01:00:00

export nodes=`cat $PBS_NODEFILE`
export nnodes=`cat $PBS_NODEFILE | wc -l`
export confile=inf.$PBS_JOBID.conf

for i in $nodes
do
  echo ${i}>>$confile
done

time python3 $HOME/Conway_Maxwell_Hierarchical_Model/py/save_cell_info.py -s
time python3 $HOME/Conway_Maxwell_Hierarchical_Model/py/save_cell_info.py -s -p frontal
