"""
Script for loading in all all the functions. Testing that loading is working.

python3 -i py/loading_script.py -d -n 2000 -b 0.005 -w 100 -s 5 
"""
import argparse, sys, os, shutil, h5py, glob
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from scipy.stats import binom, betabinom
from scipy.optimize import minimize
from multiprocessing import Pool

parser = argparse.ArgumentParser(description='For loading in the functions and loading the cell info.')
parser.add_argument('-n', '--num_cells', help='Number of cells to use.', default=100, type=int)
parser.add_argument('-b', '--bin_width', help='Time bin with to use (in seconds).', default=0.001, type=float)
parser.add_argument('-r', '--region', help='The region to use for any ad hoc plotting.', default='thalamus', type=str)
parser.add_argument('-w', '--window_size', help='The number of bins to use for fitting.', default=100, type=int)
parser.add_argument('-s', '--window_skip', help='The number of bins between fitting windows.', default=10, type=int)
parser.add_argument('-d', '--debug', help='Enter debug mode.', default=False, action='store_true')
args = parser.parse_args()

pd.set_option('max_rows', 30)
np.set_printoptions(linewidth=shutil.get_terminal_size().columns)

proj_dir = os.path.join(os.environ['PROJ'], 'Conway_Maxwell_Hierarchical_Model')
csv_dir = os.path.join(proj_dir, 'csv')
mat_dir = os.path.join(proj_dir, 'mat')
py_dir = os.path.join(proj_dir, 'py')
h5_dir = os.path.join(proj_dir, 'h5')
image_dir = os.path.join(proj_dir, 'images')
posterior_dir = os.path.join(proj_dir, 'posterior')
frontal_dir = os.path.join(proj_dir, 'frontal')

sys.path.append(py_dir)
sys.path.append(os.path.join(os.environ['PROJ'], 'Conway_Maxwell_Binomial_Distribution'))
import ConwayMaxwellHierarchicalModel as comh
import ConwayMaxwellBinomial as comb

print(dt.datetime.now().isoformat() + ' INFO: ' + 'Starting main function...')
cell_info = comh.loadCellInfo(csv_dir)
stim_info, stim_ids = comh.loadStimulusInfo(mat_dir)
adj_cell_ids = comh.getRandomSubsetOfCells(cell_info, args.num_cells)
bin_width, window_size = [args.bin_width, args.window_size]
h5_file_list = comh.getFileListFromTrialIndices(h5_dir, stim_info[stim_info['stim_ids'] != 17].index.values, args.bin_width, args.window_size)
title = args.region.capitalize().replace('_', ' ') + ', Num Trials=' + str(len(h5_file_list)) + ' All stimulated trials'
trial_info = stim_info.loc[comh.getTrialIndexFromH5File(h5py.File(h5_file_list[0],'r'))]
stim_times = [trial_info['stim_starts'], trial_info['stim_stops']]
plt.figure(figsize=(10,4))
comh.plotAverageMeasure(h5_file_list, args.region, 'moving_avg', index=None, stim_times=stim_times, label='Moving Avg.', title=title)
save_dir = os.path.join(image_dir, 'Averaging_measurements_across_trials', args.region, str(int(1000*args.bin_width)) + 'ms', 'moving_avg')
os.makedirs(save_dir) if not os.path.exists(save_dir) else None
save_name = os.path.join(save_dir, 'moving_avg_all_stimulated_trials.png')
plt.savefig(save_name)
plt.close('all')
plt.figure(figsize=(10,4))
comh.plotAverageMeasure(h5_file_list, args.region, 'binom_params', index=None, stim_times=stim_times, label=r'Binomial $p$', title=title)
save_dir = os.path.join(image_dir, 'Averaging_measurements_across_trials', args.region, str(int(1000*args.bin_width)) + 'ms', 'binom_params')
os.makedirs(save_dir) if not os.path.exists(save_dir) else None
save_name = os.path.join(save_dir, 'binom_p_all_stimulated_trials.png')
plt.savefig(save_name)
plt.close('all')
plt.figure(figsize=(10,4))
comh.plotAverageMeasure(h5_file_list, args.region, 'comb_params', index=0, stim_times=stim_times, label=r'COM-Binomial $p$', title=title)
save_dir = os.path.join(image_dir, 'Averaging_measurements_across_trials', args.region, str(int(1000*args.bin_width)) + 'ms', 'comb_p')
os.makedirs(save_dir) if not os.path.exists(save_dir) else None
save_name = os.path.join(save_dir, 'comb_p_all_stimulated_trials.png')
plt.savefig(save_name)
plt.close('all')
plt.figure(figsize=(10,4))
comh.plotAverageMeasure(h5_file_list, args.region, 'comb_params', index=1, stim_times=stim_times, label=r'COM-Binomial $\nu$', title=title)
save_dir = os.path.join(image_dir, 'Averaging_measurements_across_trials', args.region, str(int(1000*args.bin_width)) + 'ms', 'comb_nu')
os.makedirs(save_dir) if not os.path.exists(save_dir) else None
save_name = os.path.join(save_dir, 'comb_nu_all_stimulated_trials.png')
plt.savefig(save_name)
plt.close('all')
plt.figure(figsize=(10,4))
comh.plotAverageMeasure(h5_file_list, args.region, 'betabinom_ab', index=0, stim_times=stim_times, label=r'Beta-Binomial $\pi$', reparametrise=True, title=title)
save_dir = os.path.join(image_dir, 'Averaging_measurements_across_trials', args.region, str(int(1000*args.bin_width)) + 'ms', 'betabinom_pi')
os.makedirs(save_dir) if not os.path.exists(save_dir) else None
save_name = os.path.join(save_dir, 'betabinom_pi_all_stimulated_trials.png')
plt.savefig(save_name)
plt.close('all')
plt.figure(figsize=(10,4))
comh.plotAverageMeasure(h5_file_list, args.region, 'betabinom_ab', index=1, stim_times=stim_times, label=r'Beta-Binomial $\rho$', reparametrise=True, title=title)
save_dir = os.path.join(image_dir, 'Averaging_measurements_across_trials', args.region, str(int(1000*args.bin_width)) + 'ms', 'betabinom_rho')
os.makedirs(save_dir) if not os.path.exists(save_dir) else None
save_name = os.path.join(save_dir, 'betabinom_rho_all_stimulated_trials.png')
plt.savefig(save_name)
plt.close('all')


# TODO Consolidate into a 4 loop running over a few 5 element lists
