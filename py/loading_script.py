"""
Script for loading in all all the functions. Testing that loading is working.

python3 -i py/loading_script.py -d -n 2000 -b 0.005 -w 100 -s 5 
"""
import argparse, sys, os, shutil, h5py, glob
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from scipy.stats import binom, betabinom, mannwhitneyu
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
posterior_dir = os.path.join(proj_dir, 'posterior')
frontal_dir = os.path.join(proj_dir, 'frontal')
image_dir = os.path.join(proj_dir, 'images')

sys.path.append(py_dir)
sys.path.append(os.path.join(os.environ['PROJ'], 'Conway_Maxwell_Binomial_Distribution'))
import ConwayMaxwellHierarchicalModel as comh
import ConwayMaxwellBinomial as comb

print(dt.datetime.now().isoformat() + ' INFO: ' + 'Starting main function...')
cell_info = comh.loadCellInfo(csv_dir)
stim_info, stim_ids = comh.loadStimulusInfo(mat_dir)
h5_file_list = comh.getFileListFromTrialIndices(h5_dir, stim_info.index.values, args.bin_width, args.window_size)
best_ind, binom_best_ll, betabinom_best_ll, comb_best_ll, binom_best_params, betabinom_best_params, comb_best_params, num_active_cells_binned, num_cells = comh.getExampleForFileRegion(h5_file_list[20], args.region)

bin_dist = binom(num_cells, binom_best_params)
betabin_dist = betabinom(num_cells, betabinom_best_params[0], betabinom_best_params[1])
comb_dist = comb.ConwayMaxwellBinomial(comb_best_params[0], comb_best_params[1], num_cells)
title = args.region.capitalize() + ', ' + str(num_cells) + ' cells, ' + str(args.window_size) + 'ms window'
comh.plotCompareDataFittedDistnBar(num_active_cells_binned, [bin_dist, betabin_dist, comb_dist], data_label='Empirical Distn', distn_label=['Binomial PMF', 'Beta-binomial PMF', 'COM-binomial PMF'], title=title, colours=['blue', 'orange', 'green'])
