"""
Script for loading in all all the functions. Testing that loading is working.

python3 -i py/loading_script.py -d
"""
import argparse, sys, os, shutil
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='For loading in the functions and loading the cell info.')
parser.add_argument('-n', '--num_cells', help='Number of cells to use.', default=100, type=int)
parser.add_argument('-d', '--debug', help='Enter debug mode.', default=False, action='store_true')
args = parser.parse_args()

pd.set_option('max_rows', 30)
np.set_printoptions(linewidth=shutil.get_terminal_size().columns)

proj_dir = os.path.join(os.environ['PROJ'], 'Conway_Maxwell_Hierarchical_Model')
csv_dir = os.path.join(proj_dir, 'csv')
mat_dir = os.path.join(proj_dir, 'mat')
py_dir = os.path.join(proj_dir, 'py')
posterior_dir = os.path.join(proj_dir, 'posterior')
frontal_dir = os.path.join(proj_dir, 'frontal')

sys.path.append(py_dir)
import ConwayMaxwellHierarchicalModel as comh

if not args.debug:
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Starting main function...')
    cell_info = comh.loadCellInfo(csv_dir)
    stim_info, stim_ids = comh.loadStimulusInfo(mat_dir)
    adj_cell_ids = comh.getRandomSubsetOfCells(cell_info, args.num_cells)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading spike time dictionary...')
    spike_time_dict = comh.loadSpikeTimeDict(adj_cell_ids, posterior_dir, frontal_dir, cell_info)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loaded.')
    interval_start_time = np.mean([stim_info.loc[0]['stim_stops'], stim_info.loc[1]['stim_starts']])
    interval_end_time = np.mean([stim_info.loc[1]['stim_stops'], stim_info.loc[2]['stim_starts']])
    stim_start_time = stim_info.loc[1]['stim_starts']
    stim_end_time = stim_info.loc[1]['stim_stops']
    bin_width = 0.001
    bin_borders, num_active_cells_binned = comh.getNumberOfActiveCellsInBinnedInterval(interval_start_time, interval_end_time, bin_width, spike_time_dict)
