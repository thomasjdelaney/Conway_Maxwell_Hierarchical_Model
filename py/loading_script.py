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
posterior_dir = os.path.join(proj_dir, 'posterior')
frontal_dir = os.path.join(proj_dir, 'frontal')

sys.path.append(py_dir)
sys.path.append(os.path.join(os.environ['PROJ'], 'Conway_Maxwell_Binomial_Distribution'))
import ConwayMaxwellHierarchicalModel as comh
import ConwayMaxwellBinomial as comb

print(dt.datetime.now().isoformat() + ' INFO: ' + 'Starting main function...')
cell_info = comh.loadCellInfo(csv_dir)
h5_file_list = comh.getFileListFromTrialIndices(h5_dir, list(range(170)), bin_width=0.001, window_size=100)
regions=['thalamus','v1','striatum','motor_cortex','hippocampus']
relevant_h5_files = []
relevant_regions = []
for h5_file_name in h5_file_list:
    h5_file = h5py.File(h5_file_name, 'r')
    for region in regions:
        binom_log_like = h5_file.get(region).get('binom_log_like')
        betabinom_log_like = h5_file.get(region).get('betabinom_log_like')
        comb_log_like = h5_file.get(region).get('comb_log_like')
        winning_distns_for_region_file = np.argmax([binom_log_like, betabinom_log_like, comb_log_like], axis=0)
        if np.any(winning_distns_for_region_file != 2):
            relevant_h5_files.append(h5_file_name)
            relevant_regions.append(region)
            
