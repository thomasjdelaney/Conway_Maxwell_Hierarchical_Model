#!/bin/bash

proj_dir=$HOME/Conway_Maxwell_Hierarchical_Model
py_dir=$proj_dir/py

bin_width=$1
window_size=$2
trial_summary_flag=$3

python3 $py_dir/plotting_script.py -r thalamus -b $bin_width -w $window_size $trial_summary_flag
python3 $py_dir/plotting_script.py -r v1 -b $bin_width -w $window_size $trial_summary_flag
python3 $py_dir/plotting_script.py -r motor_cortex -b $bin_width -w $window_size $trial_summary_flag
python3 $py_dir/plotting_script.py -r striatum -b $bin_width -w $window_size $trial_summary_flag
python3 $py_dir/plotting_script.py -r hippocampus -b $bin_width -w $window_size $trial_summary_flag
