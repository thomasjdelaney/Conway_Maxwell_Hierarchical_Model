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

def plotAverageMeasure(h5_file_list, region, measure, index=None, include_stim=True, colour='blue', **kwargs):
    """
    For plotting the average of a given measure taking the average across the given files.
    Arguments:  h5_file_list, list(str), file_names
                region, str
                measure, str, for example 'moving_avg', 'comb_params'
                index, int, for indexing into two parameter columns
                include_stim, include the stimulus shaded area
    Returns:    None
    """
    x_axis = h5py.File(h5_file_list[0],'r').get('window_centre_times')[()]
    time_adjustor = x_axis[0]
    x_axis = x_axis - time_adjustor
    measures = []
    for h5_file_name in h5_file_list:
        h5_file = h5py.File(h5_file_name, 'r')
        trial_measure = h5_file.get(region).get(measure)[()]
        trial_measure = trial_measure[:,index] if index != None else trial_measure
        measures.append(trial_measure)
        plt.plot(x_axis, trial_measure, color=colour, alpha=0.05)
        h5_file.close()
    plt.plot(x_axis, np.array(measures).mean(axis=0), color=colour, **kwargs)
    trial_index = comh.getTrialIndexFromH5File(h5py.File(h5_file_list[0],'r'))
    if include_stim:
        comh.plotShadedStimulus([stim_info.loc[trial_index]['stim_starts']]-time_adjustor, [stim_info.loc[trial_index]['stim_stops']]-time_adjustor, plt.ylim()[1])
    plt.legend(fontsize='large') if 'label' in kwargs else None
    plt.xlim((x_axis[0], x_axis[-1]))
    plt.ylim((0, np.max(measures)))
    plt.xlabel('Time (s)', fontsize='large')
    return None

print(dt.datetime.now().isoformat() + ' INFO: ' + 'Starting main function...')
cell_info = comh.loadCellInfo(csv_dir)
stim_info, stim_ids = comh.loadStimulusInfo(mat_dir)
adj_cell_ids = comh.getRandomSubsetOfCells(cell_info, args.num_cells)
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading spike time dictionary...')
spike_time_dict = comh.loadSpikeTimeDict(adj_cell_ids, posterior_dir, frontal_dir, cell_info)
region_to_spike_time_dict = comh.divideSpikeTimeDictByRegion(spike_time_dict,cell_info)
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loaded.')
bin_width, window_size = [args.bin_width, args.window_size]
region=args.region
h5_file_list = glob.glob(os.path.join(h5_dir, '*_bin_width_' + str(int(1000*bin_width)) + 'ms_num_bins_' + str(window_size) + '.h5'))
plt.figure()
plotAverageMeasure(h5_file_list, args.region, 'moving_avg', index=None, include_stim=True, label='Moving Avg.')
plt.figure()
plotAverageMeasure(h5_file_list, args.region, 'binom_params', index=None, include_stim=True, label=r'Binomial $p$')
plt.figure()
plotAverageMeasure(h5_file_list, args.region, 'comb_params', index=0, include_stim=True, label=r'COM-Binomial $p$')
plt.figure()
plotAverageMeasure(h5_file_list, args.region, 'comb_params', index=1, include_stim=True, label=r'COM-Binomial $\nu$')
plt.show(block=False)

# need a 'get file list from trial indices' function.
# need a quick way to measure correlations :( 

###############################################################################
##################### DEMO STARTS HERE ########################################
###############################################################################

if not args.debug:
    interval_start_time = stim_info.loc[0]['stim_starts'] - 0.5
    interval_end_time = stim_info.loc[2]['stim_stops'] + 0.5
    bin_width = args.bin_width
    bin_borders, region_to_active_cells = comh.getNumberOfActiveCellsByRegion(interval_start_time, interval_end_time, bin_width, region_to_spike_time_dict)
    
    num_active_cells_binned = region_to_active_cells.get(args.region)
    total_cells = len(region_to_spike_time_dict.get(args.region))
    fitted_binom = comh.fitBinomialDistn(num_active_cells_binned, total_cells)
    fitted_binom_log_like = fitted_binom.logpmf(num_active_cells_binned).sum()
    fitted_betabinom = comh.easyLogLikeFit(betabinom, num_active_cells_binned, [1.0, 1.0], [(np.finfo(float).resolution, None), (np.finfo(float).resolution, None)], total_cells)
    fitted_betabinom_log_like = fitted_betabinom.logpmf(num_active_cells_binned).sum()
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Fitting Conway-Maxwell-binomial distribution...')
    fitted_comb_params = comb.estimateParams(total_cells, num_active_cells_binned, [0.5, 1.0])
    fitted_comb = comb.ConwayMaxwellBinomial(fitted_comb_params[0], fitted_comb_params[1], total_cells)
    fitted_comb_log_like = -comb.conwayMaxwellNegLogLike(fitted_comb_params, total_cells, num_active_cells_binned)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Fitted.')
    plt.figure(figsize=(5,4))
    comh.plotNumActiveCellsByTimeByRegion(bin_borders, {args.region.capitalize():region_to_active_cells.get(args.region)}, stim_starts=stim_info.stim_starts[:3].values, stim_stops=stim_info.stim_stops[:3].values)
    comh.plotNumActiveCellsByTimeByRegion(bin_borders[50:-49], {'Moving avg.':comh.movingAverage(region_to_active_cells.get(args.region),n=100)})
    plt.figure()
    comh.plotCompareDataFittedDistn(num_active_cells_binned, [fitted_binom, fitted_betabinom, fitted_comb], distn_label=['Binomial PMF', 'Beta-Binomial PMF', 'COM-Binomial PMF'], title='Fitted distns, region = ' + args.region.capitalize())
    plt.show(block=False)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Binomial log likelihood: ' + str(fitted_binom_log_like.round(2)))
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Beta-binomial log likelihood: ' + str(fitted_betabinom_log_like.round(2)))
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Conway-Maxwell-Binomial log likelihood: ' + str(fitted_comb_log_like.round(2)))

    spike_time_dict, bin_width, region, read_start, read_stop, stim_start, stim_stop, stim_id = region_to_spike_time_dict.get('thalamus'), 0.001, 'thalamus', stim_info.loc[0,'read_starts'], stim_info.loc[0,'read_stops'], stim_info.loc[0,'stim_starts'], stim_info.loc[0,'stim_stops'], stim_info.loc[0,'stim_ids']

##############################################################################
################## PROCESSING h5 FILES #######################################
##############################################################################

# TODO  Function for rolling over 100ms windows.
#       Save as hdf5 files. One per stimulus per bin_width value.
