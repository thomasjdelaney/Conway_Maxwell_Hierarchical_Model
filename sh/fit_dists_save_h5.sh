#!/bin/bash
#
# request resources:
#PBS -l nodes=1:ppn=16,walltime=15:00:00

# on compute node, change directory to 'submission directory':

export nodes=`cat $PBS_NODEFILE`
export nnodes=`cat $PBS_NODEFILE | wc -l`
export confile=inf.$PBS_JOBID.conf

for i in $nodes
do
  echo ${i}>>$confile
done

# run your program, timing it for good measure:
# NB: Only using 50 cells here because of memory constraints
time python3 -Wi $HOME/Conway_Maxwell_Hierarchical_Model/py/saving_hdf5_script.py -n 2000 -b 0.001 
# time python3 -Wi $HOME/Conway_Maxwell_Hierarchical_Model/py/saving_hdf5_script.py -n 2000 -b 0.005 -w 40 -s 2 
# time python3 -Wi $HOME/Conway_Maxwell_Hierarchical_Model/py/saving_hdf5_script.py -n 2000 -b 0.01 -w 40 -s 1 
