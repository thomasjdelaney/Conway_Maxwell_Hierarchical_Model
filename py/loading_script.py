"""
Script for loading in all all the functions. Testing that loading is working.

python3 -i py/loading_script.py -d
"""
import argparse, sys, os, shutil
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from scipy.stats import nbinom, binom, betabinom
from scipy.optimize import minimize
from scipy.special import gammaln
from multiprocessing import Pool

parser = argparse.ArgumentParser(description='For loading in the functions and loading the cell info.')
parser.add_argument('-n', '--num_cells', help='Number of cells to use.', default=100, type=int)
parser.add_argument('-b', '--bin_width', help='Time bin with to use (in seconds).', default=0.001, type=float)
parser.add_argument('-r', '--region', help='The region to use for any ad hoc plotting.', default='thalamus', type=str)
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
sys.path.append(os.path.join(os.environ['PROJ'], 'Conway_Maxwell_Binomial_Distribution'))
import ConwayMaxwellHierarchicalModel as comh
import ConwayMaxwellBinomial as comb

def getTrialMeasurements(spike_time_dict, bin_width, region, read_start, read_stop, stim_start, stim_stop, stim_id, num_bins_fitting=100):
    """
    Get all the measurements we want for a trial, providing start and stop times, and id.
    Arguments:  spike_time_dict, dictionary adj_cell_id => spike times
                bin_width, float,
                region, NB could be 'all'
                read_start, start counting the activity here
                read_stop, stop counting at this time
                stim_start, the stimulus starts here
                stim_stop, the stimulus stops here
                stim_id, the stimulus id
                num_bins_fitting, int, the number of bins we will use to fit the distributions
    Returns:    num_active_cells_binned,
                bin_width,
                moving_avg, 
                binom_params, 
                betabinom_params, 
                comb_params,
                binom_log_like,
                betabinom_log_like,
                comb_log_like.
    """
    num_cells = len(spike_time_dict)
    bin_borders, num_active_cells_binned = comh.getNumberOfActiveCellsInBinnedInterval(read_start, read_stop, bin_width, spike_time_dict)
    rolling_array = np.zeros([num_bins_fitting, num_active_cells_binned.size + num_bins_fitting], dtype=int)
    for i in range(num_bins_fitting): # create the windows to fit on before fitting starts.
        rolling_array[i,(num_bins_fitting - i):(num_bins_fitting - i + num_active_cells_binned.size)] = num_active_cells_binned
    fitting_counts = rolling_array[:,range(num_bins_fitting, num_active_cells_binned.size)]
    num_counts_to_fit = fitting_counts.shape[1]
    moving_avg = fitting_counts.mean(axis=0)
    with Pool() as pool:
        binom_params_future = pool.starmap_async(comh.fitBinomialDistn, zip(fitting_counts.T, [num_cells]*num_counts_to_fit))
        betabinom_params_future = pool.starmap_async(comh.easyLogLikeFit, zip([betabinom]*num_counts_to_fit, fitting_counts.T, [[1.0,1.0]]*num_counts_to_fit, [((1e-08,None),(1e-08,None))]*num_counts_to_fit, [num_cells]*num_counts_to_fit))
        comb_params_future = pool.starmap_async(comb.estimateParams, zip([num_cells]*num_counts_to_fit, fitting_counts.T, [[0.5, 1.0]]*num_counts_to_fit))
        binom_params_future.wait()
        betabinom_params_future.wait()
        comb_params_future.wait()
    binom_params = np.array([b.args[1] for b in binom_params_future.get()])
    betabinom_ab = np.array([[b.kwds['a'],b.kwds['b']] for b in betabinom_params_future.get()])
    betabinom_pr = np.array([[alpha/(alpha+beta),1/(alpha+beta+1)] for alpha,beta in betabinom_ab])
    comb_params = np.array(comb_params_future.get())
    binom_log_like = np.array([binom.logpmf(fc, num_cells, p).sum() for fc,p in zip(fitting_counts.T,binom_params)])
    betabinom_log_like = np.array([betabinom.logpmf(fc, num_cells, p[0], p[1]).sum() for fc,p in zip(fitting_counts.T, betabinom_ab)])
    comb_log_like = -np.array([comb.conwayMaxwellNegLogLike(p, num_cells, fc) for fc,p in zip(fitting_counts.T, comb_params)])
    return num_active_cells_binned, moving_avg, binom_params, binom_log_like, betabinom_ab, betabinom_log_like, comb_params, comb_log_like

if not args.debug:
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Starting main function...')
    cell_info = comh.loadCellInfo(csv_dir)
    stim_info, stim_ids = comh.loadStimulusInfo(mat_dir)
    adj_cell_ids = comh.getRandomSubsetOfCells(cell_info, args.num_cells)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading spike time dictionary...')
    spike_time_dict = comh.loadSpikeTimeDict(adj_cell_ids, posterior_dir, frontal_dir, cell_info)
    region_to_spike_time_dict = comh.divideSpikeTimeDictByRegion(spike_time_dict,cell_info)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loaded.')
    
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

    num_active_cells_binned, moving_avg, binom_params, binom_log_like, betabinom_ab, betabinom_log_like, comb_params, comb_log_like = getTrialMeasurements(spike_time_dict, bin_width, region, read_start, read_stop, stim_start, stim_stop, stim_id)

# TODO  Function for rolling over 100ms windows.
#       Save as hdf5 files. One per stimulus per bin_width value.
