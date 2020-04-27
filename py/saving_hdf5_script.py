"""
Script for loading in all all the functions. Testing that loading is working.

python3 -i py/loading_script.py -d -n 2000 -b 0.005 -w 100 -s 5 
"""
import argparse, sys, os, shutil, h5py
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

def saveMeasurementsForAllTrials(bin_width, stim_info, region_to_spike_time_dict, h5_dir, window_size=100, window_skip=10):
    """
    Get the measurements for each trial and save them down, one by one. 
    Arguments:  bin_width, float,
                stim_info, pandas DataFrame
    Returns:    nothing
    """
    region_to_num_cells = {r:len(d)for r,d in region_to_spike_time_dict.items()}
    for trial_index in stim_info.index.values:
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing trial number ' + str(trial_index) + '...')
        trial_bin_width_file_name = comh.getH5FileName(h5_dir, trial_index, bin_width, window_size)
        if os.path.isfile(trial_bin_width_file_name):
            print(dt.datetime.now().isoformat() + ' INFO: ' + 'Already have this file. Skipping...')
            continue
        trial_bin_width_file = h5py.File(trial_bin_width_file_name, 'w')
        trial_info = stim_info.loc[trial_index]
        bin_borders, region_to_active_cells, region_to_spike_counts = comh.getNumberOfActiveCellsByRegion(trial_info['read_starts'], trial_info['read_stops'], bin_width, region_to_spike_time_dict)
        is_stimulated = comh.isStimulatedBins(bin_borders, trial_info['stim_starts'], trial_info['stim_stops'])
        bin_centres = comh.getBinCentres(bin_borders)
        num_bins = bin_centres.size
        window_starts = np.arange(0, num_bins-window_size, window_skip)
        window_centre_times = bin_centres[window_starts+(window_size//2)]
        window_inds = np.vstack([ws + np.arange(window_size) for ws in window_starts])
        trial_bin_width_file.create_dataset('bin_width',data=bin_width)
        trial_bin_width_file.create_dataset('window_size',data=window_size)
        trial_bin_width_file.create_dataset('window_skip',data=window_skip)
        trial_bin_width_file.create_dataset('window_centre_times',data=window_centre_times)
        for region in region_to_active_cells.keys():
            regional_active_cells_binned = region_to_active_cells.get(region)
            regional_spike_count_array = region_to_spike_counts.get(region)
            print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing region ' + region + '...')
            moving_avg, corr_avg, all_stimulated, any_stimulated, binom_params, binom_log_like, betabinom_ab, betabinom_log_like, comb_params, comb_log_like = comh.getTrialMeasurements(regional_active_cells_binned, is_stimulated, window_inds, region_to_num_cells.get(region), window_size=100, window_skip=10)
            regional_group = trial_bin_width_file.create_group(region)
            regional_group.create_dataset('num_cells',data=region_to_num_cells.get(region))
            regional_group.create_dataset('num_active_cells_binned',data=regional_active_cells_binned)
            regional_group.create_dataset('region',data=region)
            regional_group.create_dataset('moving_avg',data=moving_avg)
            regional_group.create_dataset('corr_avg',data=corr_avg)
            regional_group.create_dataset('all_stimulated',data=all_stimulated)
            regional_group.create_dataset('any_stimulated',data=any_stimulated)
            regional_group.create_dataset('binom_params',data=binom_params)
            regional_group.create_dataset('binom_log_like',data=binom_log_like)
            regional_group.create_dataset('betabinom_ab',data=betabinom_ab)
            regional_group.create_dataset('betabinom_log_like',data=binom_log_like)
            regional_group.create_dataset('comb_params',data=comb_params)
            regional_group.create_dataset('comb_log_like',data=comb_log_like)
        trial_bin_width_file.close()
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done saving.')
    return None
        
if not args.debug:
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Starting main function...')
    cell_info = comh.loadCellInfo(csv_dir)
    stim_info, stim_ids = comh.loadStimulusInfo(mat_dir)
    adj_cell_ids = comh.getRandomSubsetOfCells(cell_info, args.num_cells)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading spike time dictionary...')
    spike_time_dict = comh.loadSpikeTimeDict(adj_cell_ids, posterior_dir, frontal_dir, cell_info)
    region_to_spike_time_dict = comh.divideSpikeTimeDictByRegion(spike_time_dict,cell_info)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loaded.')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Measuring and saving...')
    saveMeasurementsForAllTrials(args.bin_width, stim_info, region_to_spike_time_dict, h5_dir, window_size=args.window_size, window_skip=args.window_skip)

