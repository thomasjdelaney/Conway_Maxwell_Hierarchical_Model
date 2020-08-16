"""
Script for loading in all all the functions. Testing that loading is working.

python3 -i py/loading_script.py -d -n 2000 -b 0.005 -w 100 -s 5 
"""
import argparse, sys, os, shutil, h5py, glob
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import cm
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
h5_file_list = comh.getFileListFromTrialIndices(h5_dir, stim_info[stim_info['stim_ids'] != 17].index.values, args.bin_width, args.window_size)
unstimulated_log_likes, stimulated_log_likes, unstim_binom, stim_binom, unstim_betabinom, stim_betabinom, unstim_comb, stim_comb = comh.getLikelihoodsForRegion(h5_file_list, args.region)
unstim_betabinom_pi = comh.reparametriseBetaBinomial(unstim_betabinom)
stim_betabinom_pi = comh.reparametriseBetaBinomial(stim_betabinom)
comh.plotTwoParamHistogram(image_dir,'beta_binom',r'$\pi$', r'$\rho$', args.region, args.bin_width, 'unstim', unstim_betabinom_pi[:,0], unstim_betabinom_pi[:,1], title='Beta-binomial, ' + args.region + ', ' + str(len(h5_file_list)) + ' trials, unstimulated window')
comh.plotTwoParamHistogram(image_dir,'beta_binom',r'$\pi$', r'$\rho$', args.region, args.bin_width, 'stim', stim_betabinom_pi[:,0], stim_betabinom_pi[:,1], title='Beta-binomial, ' + args.region + ', ' + str(len(h5_file_list)) + ' trials, stimulated window')
