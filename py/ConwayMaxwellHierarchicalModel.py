import os, sys, h5py
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.io import loadmat
from multiprocessing import Pool
from scipy.stats import binom, betabinom, mannwhitneyu
from scipy.optimize import minimize

sys.path.append(os.path.join(os.environ['PROJ'], 'Conway_Maxwell_Binomial_Distribution'))
import ConwayMaxwellBinomial as comb

regions = ['v1', 'motor_cortex', 'thalamus', 'striatum', 'hippocampus']

def loadCellInfo(csv_dir):
    """
    For loading the csv containing information about each cell.
    Arguments:  csv_dir, the directory where the file can be found
    Returns:    pandas DataFrame
    """
    return pd.read_csv(os.path.join(csv_dir, 'cell_info.csv'), index_col='adj_cluster_id')

def getExtraTrialTime(stim_info):
    """
    For getting the extra time that we pick up before and after each trial from stimulus number 2.
    Arguments:  stim_info, pandas DataFrame
    Returns:    float
    """
    min_gap_time = np.min(stim_info['stim_starts'][1:].values - stim_info['stim_stops'][:-1].values)
    return np.round(min_gap_time/2,2)

def loadStimulusInfo(mat_dir, stim_number=2):
    """
    For loading information about the stimulus.
    Arguments:  mat_dir,
    Returns:    stim_info, pandas DataFrame
                stim_ids, unique stimulus ids, the most useful part
    """
    stim_file = 'experiment' + str(stim_number) + 'stimInfo.mat'
    stim_info_dict = loadmat(os.path.join(mat_dir, stim_file))
    stim_ids = np.unique(stim_info_dict['stimIDs'][0])
    stim_info = pd.DataFrame(data={'stim_ids':stim_info_dict['stimIDs'][0], 'stim_starts':stim_info_dict['stimStarts'][0],
        'stim_stops':stim_info_dict['stimStarts'][0]+1.5})
    stim_info['read_starts'] = stim_info['stim_starts'] - getExtraTrialTime(stim_info)
    stim_info['read_stops'] = stim_info_dict['stimStops'][0] + getExtraTrialTime(stim_info)
    return stim_info, stim_ids

def getIdAdjustor(cell_info):
    """
    For getting the integer used to adjust cluster ids, to make them all unique. frontal adjusted cluster ids = frontal cluster ids + id adjustor.
    Arguments:  cell_info, pandas dataframe.
    Returns:    integer
    """
    return cell_info[cell_info['probe'] == 'posterior']['cluster_id'].max() + 1

def getRandomSubsetOfCells(cell_info, num_cells, probes=['any'], groups=['any'], regions=['any']):
    """
    For getting a random selection of cells, spread across regions evenly. Filtering by probe, group, and region is possible.
    Arguments:  cell_info, pandas DataFrame.
                num_cells, int.
                posterior_dir, string, the directory for posterior probe data.
                frontal_dir, string, the directory for frontal probe data.
                probes, list string, for filtering by probe.
                groups, list string, for filtering by cell group ['good', 'mua', 'unsorted']
                regions, list string for filtering by region.
    Returns:    adjusted cell ids, list
    """
    relevant_cell_info = cell_info[cell_info['probe'].isin(probes)] if probes != ['any'] else cell_info
    relevant_cell_info = relevant_cell_info[relevant_cell_info['group'].isin(groups)] if groups != ['any'] else relevant_cell_info
    relevant_cell_info = relevant_cell_info[relevant_cell_info['region'].isin(regions)] if regions != ['any'] else relevant_cell_info
    if num_cells > relevant_cell_info.shape[0]:
        print(dt.datetime.now().isoformat() + ' WARN: ' + 'Returning all available cells.')
        return relevant_cell_info.index.values
    remaining_regions = relevant_cell_info.region.unique()
    num_regions = remaining_regions.size
    cells_per_region = num_cells // num_regions
    distributed_cells = np.zeros(cells_per_region * num_regions, dtype=int)
    cells_added = 0
    for region in remaining_regions:
        region_cell_info = relevant_cell_info[relevant_cell_info['region'] == region]
        num_region_cells = region_cell_info.shape[0]
        if cells_per_region > num_region_cells:
            cells_to_add = region_cell_info.index.values
        else:
            cells_to_add = np.random.choice(region_cell_info.index.values, cells_per_region, replace=False)
        distributed_cells[np.arange(cells_added, cells_added + cells_to_add.size)] = cells_to_add
        cells_added += cells_to_add.size
    more_to_add = distributed_cells.size - cells_added
    if more_to_add > 0:
        cells_so_far = np.setdiff1d(distributed_cells, 0)
        all_cells = relevant_cell_info.index.values
        cells_to_add = np.random.choice(np.setdiff1d(all_cells, cells_so_far), more_to_add, replace=False)
        distributed_cells[np.arange(cells_added, cells_added + cells_to_add.size)] = cells_to_add
    return distributed_cells

def getSpikeTimesForCell(adj_cell_id, spike_times, spike_clusters):
    """
    For returning the spike times for a given cell
    Arguments:  adj_cell_id, integer
                spike_times, all spike_times
                spike_clusters, all_spike_clusters
    Returns:    spike_times for the cell, numpy array (float)
    """
    return spike_times[spike_clusters == adj_cell_id]

def loadSpikeTimeDict(adj_cell_ids, posterior_dir, frontal_dir, cell_info):
    """
    For returning a dictionary of adj_cell_id => spike time.
    Arguments:  adj_cell_ids, the adjusted cell ids of the cells for which we want the spike times.
                posterior_dir, str, the directory
                frontal_dir, str, the directory
                all_posterior, boolean, are all the cells from the posterior probe
                all_frontal, boolean, are all the cells from the frontal probe
    Returns:    dictionary, adjusted cell ids => spike_times
    """
    frames_per_second = 30000.0
    time_correction = np.load(os.path.join(frontal_dir, 'time_correction.npy'))
    post_spike_times = np.load(os.path.join(posterior_dir, 'spike_times.npy')).flatten()/frames_per_second
    front_spike_times = np.load(os.path.join(frontal_dir, 'spike_times.npy')).flatten()/frames_per_second - time_correction[1]
    post_spike_clusters = np.load(os.path.join(posterior_dir, 'spike_clusters.npy'))
    front_spike_clusters = np.load(os.path.join(frontal_dir, 'spike_clusters.npy')) + getIdAdjustor(cell_info)
    relevant_cell_info = cell_info.loc[adj_cell_ids]
    relevant_cell_info['is_frontal'] = relevant_cell_info['probe'] == 'frontal'
    arg_spike_times = list(np.array([post_spike_times, front_spike_times])[relevant_cell_info['is_frontal'].values.astype(int)])
    arg_spike_clusters = list(np.array([post_spike_clusters, front_spike_clusters])[relevant_cell_info['is_frontal'].values.astype(int)])
    with Pool() as pool:
        spike_times_future = pool.starmap_async(getSpikeTimesForCell, zip(adj_cell_ids, arg_spike_times, arg_spike_clusters))
        spike_times_future.wait()
    return dict(zip(adj_cell_ids, spike_times_future.get()))

def divideSpikeTimeDictByRegion(spike_time_dict, cell_info):
    """
    For dividing the spike time dict into more dictionaries, one for each region.
    Arguments:  spike_time_dict, dictionary adj_cell_id => spike_times
                cell_info, pandas DataFrame
    Returns:    a dictionary of spike time dictionaries for each region from which we have cells.
    """
    relevant_cell_info = cell_info.loc[list(spike_time_dict.keys())]
    relevant_regions = relevant_cell_info.region.unique()
    region_to_spike_time_dict = {}
    for region in relevant_regions:
        relevant_adj_ids = relevant_cell_info[relevant_cell_info['region'] == region].index.values
        region_to_spike_time_dict[region] = dict(zip(relevant_adj_ids, map(spike_time_dict.get, relevant_adj_ids)))
    return region_to_spike_time_dict

def getBinBorders(start_time, end_time, bin_width):
    """
    For getting the binning borders for a time interval.
    Arguments:  start_time
                end_time
                bin_width
    Returns:    array (float)
    """
    return np.concatenate([np.arange(start_time, end_time, bin_width), [end_time]])

def binActiveTimes(spike_times, bin_borders):
    """
    For binning the given spike times into active or not active time bins.
    Arguments:  spike_times, array (float)
                bin_borders, array (float)
    Returns:    array (boolean)
    """
    spike_counts, bb = np.histogram(spike_times, bins=bin_borders)
    return spike_counts

def getNumberOfActiveCellsInBinnedInterval(start_time, end_time, bin_width, spike_time_dict):
    """
    For getting the number of active cells in an interval divided into bins.
    Arguments:  start_time, float
                end_time, float
                bin_width, float
                spike_time_dict, dictionary adj_cell_ids => spike_times
    Returns:    bin_borders, array (floats)
                number of active cells, array (ints)
    """
    num_cells = len(spike_time_dict)
    bin_borders = getBinBorders(start_time, end_time, bin_width)
    spike_times_list = list(spike_time_dict.values())
    with Pool() as pool:
        is_active_future = pool.starmap_async(binActiveTimes, zip(spike_times_list, [bin_borders]*num_cells))
        is_active_future.wait()
    spike_count_array = np.vstack(is_active_future.get())
    return bin_borders, np.sum(spike_count_array > 0, axis=0), spike_count_array

def getNumberOfActiveCellsByRegion(interval_start_time, interval_end_time, bin_width, region_to_spike_time_dict):
    """
    Get a dictionary region => number of active cells.
    Arguments:  interval_start_time,
                interval_end_time,
                bin_width,
                region_to_spike_time_dict, region => dict(adj cell id => spike times)
    Returns:    dict, region => number of active cells
                bin_borders
    """
    region_to_active_cells = {}
    region_to_spike_counts = {}
    for region,regional_spike_time_dict in region_to_spike_time_dict.items():
        bin_borders, region_to_active_cells[region], region_to_spike_counts[region] = getNumberOfActiveCellsInBinnedInterval(interval_start_time, interval_end_time, bin_width, regional_spike_time_dict)
    return bin_borders, region_to_active_cells, region_to_spike_counts

def fitBinomialDistn(num_active_cells_binned, total_cells):
    """
    For fitting a binomial distribution to the given activit data.
    Arguments:  num_active_cells_binned, array int
                total_cells, int, how many cells could spike in any given bin
    Returns:    fitted binom ditribution object.
    """
    num_trials = num_active_cells_binned.size
    total_successes = num_active_cells_binned.sum()
    return binom(total_cells, total_successes/(total_cells*num_trials))

def movingAverage(a, n=3):
    """
    Arguments:  a, numpy array
                n, the number of steps over which to take the average.
    Returns:    array, of length a.size - n + 1
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def easyLogLikeFit(distn, data, init, bounds, n):
    """
    For fitting a distribution to the data using maximum likelihood method, starting from init.
    Arguments:  distn, the distribution
                data, the data
                init, initial guess
                bounds, (min, max) pairs for each parameter
                n, the number of cells
    Returns:    fitted distribution object.
    """
    res = minimize(lambda x:-distn.logpmf(data, n=n, a=x[0], b=x[1]).sum(), init, bounds=bounds)
    fitted = distn(n=n, a=res.x[0], b=res.x[1])
    return fitted

def getAverageSpikeCountCorrelation(spike_count_array):
    """
    For getting the average of the spike count correlation from an array of spike counts.
    Arguments:  spike_count_array, numpy array (int) (num_cells, num_spikes)
    Returns:    float, the average of the spike count correlations
    """
    corr_matrix = np.corrcoef(spike_count_array)
    if np.isnan(spike_count_array).all():
        return np.nan
    else:
        return np.nanmean(corr_matrix[np.triu_indices(corr_matrix.shape[0], k=1)])

def getTrialMeasurements(num_active_cells_binned, spike_count_array, is_stimulated, window_inds, num_cells, window_size=100, window_skip=10):
    """
    Get all the measurements we want for a trial, providing start and stop times, and id.
    Arguments:  num_active_cells_binned, numpy array (int)
                spike_count_array, array (int) (num_cells, num_bins)
                is_stimulated, numpy array boolean, whether or not the stimulus was present for the entirety of each bin.
                window_inds, numpy array (int) (num_windows, window_size), indices for each window, each row is a list of indices for a window.
                num_cells, int
                window_size, int, the number of bins we will use to fit the distributions
                window_skip, int, the number of bins between each fitting window. This influences how many times we have to fit each distribution,
                    and therefore has a major effect on how long it takes to process everything.
    Returns:    moving_avg,
                corr_avg,
                binom_params,
                betabinom_params,
                comb_params,
                binom_log_like,
                betabinom_log_like,
                comb_log_like.
    """
    num_windows = window_inds.shape[0]
    windowed_active_cells = num_active_cells_binned[window_inds]
    windowed_spike_counts = spike_count_array[:,window_inds].reshape((num_windows, num_cells, window_size))
    moving_avg_activity_by_cell = (spike_count_array[:,window_inds] > 0).mean(axis=2)
    moving_avg = windowed_active_cells.mean(axis=1)
    moving_var = windowed_active_cells.var(axis=1)
    all_stimulated = is_stimulated[window_inds].all(axis=1)
    any_stimulated = is_stimulated[window_inds].any(axis=1)
    with Pool() as pool:
        binom_params_future = pool.starmap_async(fitBinomialDistn, zip(windowed_active_cells, [num_cells]*num_windows))
        betabinom_params_future = pool.starmap_async(easyLogLikeFit, zip([betabinom]*num_windows, windowed_active_cells, [[1.0,1.0]]*num_windows, [((1e-08,None),(1e-08,None))]*num_windows, [num_cells]*num_windows))
        comb_params_future = pool.starmap_async(comb.estimateParams, zip([num_cells]*num_windows, windowed_active_cells))
        corr_avg_future = pool.map_async(getAverageSpikeCountCorrelation, windowed_spike_counts)
        binom_params_future.wait()
        betabinom_params_future.wait()
        comb_params_future.wait()
    binom_params = np.array([b.args[1] for b in binom_params_future.get()])
    betabinom_ab = np.array([[b.kwds['a'],b.kwds['b']] for b in betabinom_params_future.get()])
    comb_params = np.array(comb_params_future.get())
    corr_avg = corr_avg_future.get()
    binom_log_like = np.array([binom.logpmf(fc, num_cells, p).sum() for fc,p in zip(windowed_active_cells, binom_params)])
    betabinom_log_like = np.array([betabinom.logpmf(fc, num_cells, p[0], p[1]).sum() for fc,p in zip(windowed_active_cells, betabinom_ab)])
    comb_log_like = -np.array([comb.conwayMaxwellNegLogLike(p, num_cells, fc) for fc,p in zip(windowed_active_cells, comb_params)])
    return moving_avg_activity_by_cell, moving_avg, moving_var, corr_avg, all_stimulated, any_stimulated, binom_params, binom_log_like, betabinom_ab, betabinom_log_like, comb_params, comb_log_like

def isStimulatedBins(bin_borders, stim_start, stim_stop):
    """
    Get an array of booleans indicating which bins are stimulated and which are not.
    Arguments:  bin_borders, the times where the bins start and stop in seconds
                stim_start, time,
                stim_stop, time
    Returns:    numpy array boolean
    """
    return np.array([(bin_start > stim_start) & (bin_stop < stim_stop) for bin_start,bin_stop in zip(bin_borders[:-1],bin_borders[1:])])

def getH5FileName(h5_dir, trial_index, bin_width, window_size):
    """
    For getting the name of the h5 file. Can be used for saving or loading or whatever.
    Arguments:  h5_dir, string, the directory
                trial_index, int, the index of the trial
                bin_width, float,
                window_size, number of bins used to fit the distributions.
    Returns:    string
    """
    return os.path.join(h5_dir, 'trial_' + str(trial_index) + '_bin_width_' + str(int(1000*bin_width)) + 'ms_num_bins_' + str(window_size) + '.h5')

def getFileListFromTrialIndices(h5_dir, trial_indices, bin_width=0.001, window_size=100):
    """
    For getting a list of hdf5 files given a list of trial_indices, and maybe some more info.
    Arguments:  h5_dir, str, directory
                trial_indices, list of ints,
                bin_width, float
                window_size, int
    Returns:    list of file names, list(str)
    """
    trial_indices = [trial_indices] if np.isscalar(trial_indices) else trial_indices
    return [getH5FileName(h5_dir, trial_index, bin_width, window_size) for trial_index in trial_indices]

def readH5File(h5_dir, trial_index, bin_width, window_size):
    """
    For getting a h5 file handle. We can access the data through that.
    Arguments:  h5_dir, string
                trial_index, the identifies for the trial
                bin_width, float
                window_size, int
    Returns:    HDF5 file  handle in read mode
    """
    h5_file_name = getH5FileName(h5_dir, trial_index, bin_width, window_size)
    return h5py.File(h5_file_name, 'r')

def getBinCentres(bin_borders):
    """
    Function for getting the centre of bins given the borders.
    Arguments:  bin_borders, the borders of each bin
    Returns:    bin_centres, length(bin_borders) - 1, the centre of each time bin.
    """
    return (bin_borders[:-1] + bin_borders[1:])/2

def getTrialIndexFromH5File(h5_file):
    """
    Get all the information we can from the filename.
    Arguments:  h5_file, hdf5 file object
    Returns:    trial_index, bin_width, window_size
    """
    file_name = os.path.basename(h5_file.filename)
    return int(file_name.split('_')[1])

def reparametriseBetaBinomial(ab_params):
    """
    For converting alpha beta parametrisation to the pi rho parametrisation
    Arguments:  ab_params,
    returns:    pr_params
    """
    ab_params = np.array([ab_params]) if ab_params.ndim == 1 else ab_params
    return np.array([[alpha/(alpha+beta),1/(alpha+beta+1)] for alpha,beta in ab_params])

def getFanoFactorFromFiles(h5_file_list, region, window_size):
    """
    Get the fano factors for each cell given the trials in the file list, and using the given region.
    Arguments:  h5_file_list, list of str
                region, str
                window_size, int,
    Returns:    fano factors, numpy array (float) (num cells, num windows)
    """
    moving_avg_activity_by_cell = collectMeasureFromFiles(h5_file_list, region, 'moving_avg_activity_by_cell', None, False)
    total_activity_by_cell = (window_size*moving_avg_activity_by_cell).astype(int)
    variances = np.var(total_activity_by_cell,axis=0) # taking variances across trials, dims are now (num cells, num windows)
    means = np.mean(total_activity_by_cell, axis=0) # taking means across trials, dims are now (num cells, num windows)
    fanos = variances/means
    contains_nans = np.isnan(fanos).any(axis=1) # the fano factors for these cells contain nans
    full_fanos = fanos[~contains_nans,:]
    return full_fanos

def runFanoStatTest(full_fanos, h5_file_name, region):
    """
    For running the Mann-Whitney U test on chosen unstimulated and stimulated samples  of the fano factors.
    Arguments:  full_fanos, numpy array (float)
                h5_file_name, name of a file for getting the stimulated and unstimulated indices
                region, str
    Returns:    p_value, the p-value resulting from the Mann-Whitney U test
                last_unstimulated_window_time, int
                first_all_stimulated_window_time, int
    """
    h5_file = h5py.File(h5_file_name,'r')
    window_centres = h5_file.get('window_centre_times')[()]
    last_unstimulated_window_ind = h5_file.get(region).get('any_stimulated')[()].nonzero()[0].min()-1
    first_all_stimulated_window_ind = h5_file.get(region).get('all_stimulated')[()].nonzero()[0].min()
    h5_file.close()
    last_unstimulated_fanos = full_fanos[:,last_unstimulated_window_ind]
    first_all_stimulated_fanos = full_fanos[:,first_all_stimulated_window_ind]
    mw_res = mannwhitneyu(last_unstimulated_fanos, first_all_stimulated_fanos)
    return mw_res, window_centres[last_unstimulated_window_ind], window_centres[first_all_stimulated_window_ind], last_unstimulated_window_ind, first_all_stimulated_window_ind

def binCombDkl(n,p,nu):
    """
    For calulating D_kl(Comb, bin)
    Arguments:  n, int
                p, between 0 and 1
                nu, float
    Returns:    the kl divergance
    """
    comb_dist = comb.ConwayMaxwellBinomial(p,nu,n)
    support = range(n+1)
    expected_value = np.sum([v*np.log(comb.comb(n,k)) for k,v in comb_dist.samp_des_dict.items()])
    return (nu - 1)*expected_value - np.log(comb_dist.normaliser)

def getLastUnstimulatedWindowIndFirstStimulatedWindowIndFromFile(h5_file_name, region):
    """
    For getting the last unstimulated window ind from a file
    Arguments:  h5_file_name, h5py file name
                region, string
    Returns:    last_unstimulated_window_ind, int, the index
                first_all_stimulated_window_ind, int, the index
    """
    h5_file = h5py.File(h5_file_name, 'r')
    last_unstimulated_window_ind = h5_file.get(region).get('any_stimulated')[()].nonzero()[0].min()-1
    first_all_stimulated_window_ind = h5_file.get(region).get('all_stimulated')[()].nonzero()[0].min()
    if last_unstimulated_window_ind < 0:
        print(dt.datetime.now().isoformat() + ' WARN: ' + 'Last unstimulated window index is less than 0!')
    h5_file.close()
    return last_unstimulated_window_ind, first_all_stimulated_window_ind

def getLikelihoodsForRegion(h5_file_list, region):
    """
    Get an array of log likelihoods from each of the files in the file list for the given region and distribution.
    Arguments:  h5_file_list, list of str
                region, str
    Returns:    unstimulated_log_likes, numpy array (num files, 3), binom, betabinom, and comb log likes for last unstimulated window
                stimulated_log_likes, numpy array (num_files, 3), binom, betabinom, and comb log likes for first stimulated window
    """
    unstimulated_log_likes = np.zeros((len(h5_file_list), 3)) # binom, betabinom, comb
    stimulated_log_likes = np.zeros((len(h5_file_list), 3))
    unstim_binom = np.zeros(len(h5_file_list))
    stim_binom = np.zeros(len(h5_file_list))
    unstim_betabinom = np.zeros((len(h5_file_list),2))
    stim_betabinom = np.zeros((len(h5_file_list),2))
    unstim_comb = np.zeros((len(h5_file_list),2))
    stim_comb = np.zeros((len(h5_file_list),2))
    for i,h5_file_name in enumerate(h5_file_list):
        last_unstimulated_window_ind, first_all_stimulated_window_ind = getLastUnstimulatedWindowIndFirstStimulatedWindowIndFromFile(h5_file_name, region)
        h5_file = h5py.File(h5_file_name, 'r')
        unstim_bin_log_like, stim_binom_log_like = h5_file.get(region).get('binom_log_like')[[last_unstimulated_window_ind, first_all_stimulated_window_ind]]
        unstim_betabinom_log_like, stim_betabinom_log_like = h5_file.get(region).get('betabinom_log_like')[[last_unstimulated_window_ind, first_all_stimulated_window_ind]]
        unstim_comb_log_like, stim_comb_log_like = h5_file.get(region).get('comb_log_like')[[last_unstimulated_window_ind, first_all_stimulated_window_ind]]
        unstim_binom_params, stim_binom_params = h5_file.get(region).get('binom_params')[[last_unstimulated_window_ind, first_all_stimulated_window_ind]]
        unstim_betabinom_params, stim_betabinom_params = h5_file.get(region).get('betabinom_ab')[[last_unstimulated_window_ind, first_all_stimulated_window_ind]]
        unstim_comb_params, stim_comb_params = h5_file.get(region).get('comb_params')[[last_unstimulated_window_ind, first_all_stimulated_window_ind]]
        h5_file.close()
        unstimulated_log_likes[i,:] = [unstim_bin_log_like, unstim_betabinom_log_like, unstim_comb_log_like]
        stimulated_log_likes[i,:] = [stim_binom_log_like, stim_betabinom_log_like, stim_comb_log_like]
        unstim_binom[i] = unstim_binom_params
        stim_binom[i] = stim_binom_params
        unstim_betabinom[i,:] = unstim_betabinom_params
        stim_betabinom[i,:] = stim_betabinom_params
        unstim_comb[i,:] = unstim_comb_params
        stim_comb[i,:] = stim_comb_params
    return unstimulated_log_likes, stimulated_log_likes, unstim_binom, stim_binom, unstim_betabinom, stim_betabinom, unstim_comb, stim_comb    

def getExampleForFileRegion(h5_file_name, region):
    """
    Get the window for which each distribution is the best fit. Get the distribution params as well.
    Arguments:  h5_file_name, list of strings
                region, str
    Returns:    window data
                params
    """
    h5_file = h5py.File(h5_file_name,'r')
    binom_ll = h5_file.get(region).get('binom_log_like')[()]
    betabinom_ll = h5_file.get(region).get('betabinom_log_like')[()]
    comb_ll = h5_file.get(region).get('comb_log_like')[()]

    good_inds = np.flatnonzero((comb_ll > betabinom_ll) & (comb_ll > binom_ll) & (betabinom_ll > binom_ll))
    best_ind = good_inds[comb_ll[good_inds].argmax()]
    
    binom_best_ll = h5_file.get(region).get('binom_log_like')[best_ind]
    betabinom_best_ll = h5_file.get(region).get('betabinom_log_like')[best_ind]
    comb_best_ll = h5_file.get(region).get('comb_log_like')[best_ind]
    binom_best_params = h5_file.get(region).get('binom_params')[best_ind]
    betabinom_best_params = h5_file.get(region).get('betabinom_ab')[best_ind]
    comb_best_params = h5_file.get(region).get('comb_params')[best_ind]
    num_active_cells_binned = h5_file.get(region).get('num_active_cells_binned')[()]
    num_cells = h5_file.get(region).get('num_cells')[()]
    h5_file.close()
    return best_ind, binom_best_ll, betabinom_best_ll, comb_best_ll, binom_best_params, betabinom_best_params, comb_best_params, num_active_cells_binned, num_cells

##########################################################
########## PLOTTING FUNCTIONS ############################
##########################################################

def plotShadedStimulus(stim_starts, stim_stops, upper_bound, lower_bound=0):
    """
    For plotting a shaded aread to represent the times when a stimulus was present.
    Arguments:  stim_starts, list or array, the start times of the stimuli
                stim_stops, list or array, the stop times of the stimuli
                upper_bound, lower_bound, shade between these areas.
    Returns:    nothing
    """
    for i,(stim_start, stim_stop) in enumerate(zip(stim_starts, stim_stops)):
        plt.fill_between(x=[stim_start, stim_stop], y1=upper_bound, y2=lower_bound, color='grey', alpha=0.3)
    return None

def plotNumActiveCellsByTime(bin_borders, num_active_cells_binned, stim_starts=[], stim_stops=[], get_centres=False, **kwargs):
    """
    For plotting the number of active cells across bins, optionally with stimulus time shaded.
    Arguments:  bin_borders, array (float)
                num_active_cells_binned, array (int)
                stim_starts, list or array, the start times of stimuli
                stim_stops, list or array, the stop times of stimuli
    Returns:    Nothing
    """
    bin_centres = getBinCentres(bin_borders) if get_centres else bin_borders
    if (len(stim_starts) > 0) & (len(stim_stops) > 0):
        plotShadedStimulus(stim_starts, stim_stops, num_active_cells_binned.max())
    plt.plot(bin_centres, num_active_cells_binned, **kwargs)
    return None

def plotNumActiveCellsByTimeByRegion(bin_borders, region_to_active_cells, stim_starts=[], stim_stops=[], is_tight_layout=True, get_centres=False, **kwargs):
    """
    Plot the number of active cells for each region on the same plot.
    Arguments:  bin_borders,
                region_to_active_cells, dictionary
                stim_starts, list or array,
                stim_stops, list or array,
                **kwargs
    Returns:    nothing
    """
    upper_bound = np.array(list(region_to_active_cells.values())).max()
    plotShadedStimulus(stim_starts, stim_stops, upper_bound) if (len(stim_starts) > 0) & (len(stim_stops) > 0) else None
    for region, num_active_cells_binned in region_to_active_cells.items():
        plotNumActiveCellsByTime(bin_borders, num_active_cells_binned, get_centres=get_centres, label=region.capitalize(), **kwargs)
    plt.legend(fontsize='large')
    plt.xlabel('Time (s)', fontsize='large')
    plt.ylabel('Num. Active Cells', fontsize='large')
    plt.tight_layout() if is_tight_layout else None
    return None

def plotCompareDataFittedDistn(num_active_cells_binned, fitted_distn, plot_type='pdf', data_label='Num. Active Cells', distn_label=['Fitted Distn. PMF'], title='', colours=['blue']):
    """
    For comparing a fitted distribution to some data. (pdf and/or cdf?)
    Arguments:  num_active_cells_binned, array int
                fitted_distn, distribution object or list of distn objects, needs pdf and cdf functions
                plot_type, ['pdf', 'cdf']
    Returns:    nothing
    """
    fitted_distn = [fitted_distn] if list != type(fitted_distn) else fitted_distn
    num_distns = len(fitted_distn)
    distn_label = distn_label * num_distns if len(distn_label) != num_distns else distn_label
    bin_borders = range(num_active_cells_binned.max()+1)
    plt.hist(num_active_cells_binned, bins=bin_borders, density=True, label=data_label, align='left', alpha=0.35)
    for distn, d_label, colour in zip(fitted_distn,distn_label,colours):
        plt.plot(bin_borders, distn.pmf(bin_borders), label=d_label, color=colour)
    plt.legend()
    plt.xticks(fontsize='large');plt.yticks(fontsize='large')
    plt.xlabel('Num. Active Cells', fontsize='x-large')
    plt.ylabel('P(k)', fontsize='x-large')
    plt.title(title, fontsize='large') if title != '' else None
    plt.tight_layout()
    return None

def plotCompareDataFittedDistnBar(num_active_cells_binned, fitted_distn, data_label='Num. Active Cells', distn_label=['Fitted Distn. PMF'], title='', colours=['blue']):
    """
    For comparing a fitted distribution to some data in a bar plot.
    Arguments:  num_active_cells_binned, array int
                fitted_distn, distribution object or list of distn objects, needs pdf and cdf functions
    Returns:    nothing
    """
    fitted_distn = [fitted_distn] if list != type(fitted_distn) else fitted_distn
    num_distns = len(fitted_distn)
    distn_label = distn_label * num_distns if len(distn_label) != num_distns else distn_label
    bin_borders = range(num_active_cells_binned.max()+2)
    emp_pmf, _ = np.histogram(num_active_cells_binned, bins=bin_borders, density=True)
    x_axis = np.array(bin_borders[:-1])
    distn_pmfs = np.zeros((num_distns, emp_pmf.size))
    for i,distn in enumerate(fitted_distn):
        distn_pmfs[i,:] = distn.pmf(x_axis)
    all_pmfs = np.vstack([emp_pmf, distn_pmfs])
    all_labels = np.hstack([data_label, distn_label])
    all_colours = np.hstack(['skyblue', colours])
    width=0.2
    fig, ax = plt.subplots(figsize=(5,4))
    for i in range(all_pmfs.shape[0]):
        ax.bar(x_axis + (i-2)*width, all_pmfs[i,:], width, label=all_labels[i], color=all_colours[i])
    ax.legend(fontsize='large')
    ax.set_xticks(x_axis)
    ax.tick_params(axis='both', labelsize='large')
    ax.set_xlabel('Num. Active Cells', fontsize='x-large')
    ax.set_ylabel('P(k)', fontsize='x-large')
    ax.set_title(title, fontsize='large') if title != '' else None
    plt.tight_layout()
    return None

def plotTrialSummary(h5_file, region, stim_info):
    """
    Make some plots giving a summary of a trial.
    Arguments:  h5_file, the file that contains the information that we need.
                stim_info, pandas DataFrame, information about each trial
    Returns:    nothing
    """
    trial_index = getTrialIndexFromH5File(h5_file)
    trial_info = stim_info.loc[trial_index]
    bin_borders = getBinBorders(trial_info.read_starts, trial_info.read_stops, h5_file.get('bin_width')[()]) - trial_info.stim_starts
    window_centres = h5_file.get('window_centre_times')[()] - trial_info.stim_starts
    stim_starts = 0.0; stim_stops = trial_info.stim_stops - trial_info.stim_starts;
    fig=plt.figure(figsize=(12,10))
    plt.subplot(2,2,1)
    plotNumActiveCellsByTimeByRegion(bin_borders, {'Num. active cells':h5_file.get(region).get('num_active_cells_binned')[()]}, stim_starts=[stim_starts], stim_stops=[stim_stops], get_centres=True)
    plt.plot(window_centres, h5_file.get(region).get('moving_avg')[()], label='Moving Average')
    plt.title(region.capitalize().replace('_', ' ') + ', Trial ' + str(trial_index) + ', Total cells = ' + str(h5_file.get(region).get('num_cells')[()]), fontsize='large')
    ax1 = plt.subplot(2,2,2)
    plt.plot(window_centres, h5_file.get(region).get('binom_params')[()], label=r'Binom. $p$', color='blue')
    betabinom_pr = reparametriseBetaBinomial(h5_file.get(region).get('betabinom_ab'))
    plt.plot(window_centres, betabinom_pr[:,0], label=r'Beta-Binom. $\pi$', color='orange')
    plotShadedStimulus(stim_starts=[stim_starts], stim_stops=[stim_stops], upper_bound=plt.ylim()[1])
    plt.xlabel('Time (s)', fontsize='large')
    plt.legend()
    ax2 = ax1.twinx()
    ax2.plot(window_centres, h5_file.get(region).get('comb_params')[()][:,0], label=r'COM-Binom. $p$', color='green')
    plt.legend(fontsize='small')
    plt.subplot(2,2,3)
    plt.plot(window_centres, betabinom_pr[:,1], label=r'Beta-Binom. $\rho$')
    plotShadedStimulus(stim_starts=[stim_starts], stim_stops=[stim_stops], upper_bound=plt.ylim()[1])
    plt.xlabel('Time (s)', fontsize='large')
    plt.legend(fontsize='large')
    plt.subplot(2,2,4)
    plt.plot(window_centres, h5_file.get(region).get('comb_params')[()][:,1], label=r'COM-Binom. $\nu$')
    plotShadedStimulus(stim_starts=[stim_starts], stim_stops=[stim_stops], upper_bound=plt.ylim()[1], lower_bound=plt.ylim()[0])
    plt.xlabel('Time (s)', fontsize='large')
    plt.legend(fontsize='large')
    plt.tight_layout()
    return None

def getXAxisTimeAdjustor(h5_file_name, stim_start=None):
    """
    For getting the time axis and time adjustor given a file.
    Arguments:  h5_file_name, string
    Returns:    x_axis, numpy array (float) (num_windows)
                time_adjustor, float
    """
    x_axis = h5py.File(h5_file_name,'r').get('window_centre_times')[()]
    time_adjustor = stim_start if stim_start != None else x_axis[0]
    return x_axis - time_adjustor, time_adjustor

def collectMeasureFromFiles(h5_file_list, region, measure, index, reparametrise):
    """
    For collecting a given measure from each of the files in the list.
    Arguments:  h5_file_list, list of strings
                region, string
                measure, string
                index, should we index into the measure, or not.
    Returns:    array of the measures
    """
    measures = []
    for h5_file_name in h5_file_list:
        h5_file = h5py.File(h5_file_name, 'r')
        trial_measure = h5_file.get(region).get(measure)[()]
        if reparametrise & (measure=='betabinom_ab'):
            trial_measure = reparametriseBetaBinomial(trial_measure)
        trial_measure = trial_measure[:,index] if index != None else trial_measure
        measures.append(trial_measure)
        h5_file.close()
    return np.array(measures)

def plotAverageMeasure(h5_file_list, region, measure, index=None, stim_times=[], colour='blue', reparametrise=False, title='', y_label='', **kwargs):
    """
    For plotting the average of a given measure taking the average across the given files.
    Arguments:  h5_file_list, list(str), file_names
                region, str
                measure, str, for example 'moving_avg', 'comb_params'
                index, int, for indexing into two parameter columns
                stim_times, [stim_start, stim_stop]
                colour, str
                reparametrise, bool, whether to reparametrise the betabinomial paramters or not.
    Returns:    None
    """
    x_axis, time_adjustor = getXAxisTimeAdjustor(h5_file_list[0]) if stim_times==[] else getXAxisTimeAdjustor(h5_file_list[0], stim_start=stim_times[0])
    num_cells = h5py.File(h5_file_list[0],'r').get(region).get('num_cells')[()]
    measures = collectMeasureFromFiles(h5_file_list, region, measure, index, reparametrise)
    measures_mean = np.nanmean(measures, axis=0)
    measures_std = np.nanstd(measures, axis=0)
    measures_samples = np.sum(~np.isnan(measures), axis=0)
    measures_stderr = measures_std/np.sqrt(measures_samples)
    lower_bound = np.nanmin(np.append(measures_mean - measures_stderr,0))
    upper_bound = np.nanmax(measures_mean + measures_stderr)
    if stim_times != []: # include the grey shaded area to indicate the stimulus
        plotShadedStimulus([0.0], [stim_times[1] - stim_times[0]], upper_bound, lower_bound=lower_bound)
    plt.plot(x_axis, measures_mean, color=colour, **kwargs)
    plt.fill_between(x_axis, measures_mean-measures_stderr, measures_mean+measures_stderr, alpha=0.3, color=colour, label='Std. Err.')
    plt.ylim(lower_bound, upper_bound)
    plt.xlim((x_axis[0], x_axis[-1]))
    plt.xticks(fontsize='large');plt.yticks(fontsize='large')
    plt.xlabel('Time (s)', fontsize='x-large')
    plt.ylabel(y_label, fontsize='x-large') if y_label != '' else None
    plt.title(title + ', Num cells = '+str(num_cells), fontsize='large') if title != '' else None
    plt.legend(fontsize='large') if 'label' in kwargs else None
    plt.tight_layout()
    return None

def plotCellFanoFactors(h5_file_list, region, stim_times=[], colour='blue', is_tight_layout=True, use_title=False, window_size=100):
    """
    For plotting the Fano factor for active cells, and the mean fano factor across cells. Making this to compare to Churchland et al. 2010.
        'Stimulus onset quenches neural variability: a widespread cortical phenomenon'
    Arguments:  h5_file_list, list of strings
                region, string
                stim_times, the stimulus,
                colour, str,
                is_tight_layout, use the tight layout
                use_title,
                window_size, int
    Return:     Nothing
    """
    x_axis, time_adjustor = getXAxisTimeAdjustor(h5_file_list[0]) if stim_times==[] else getXAxisTimeAdjustor(h5_file_list[0], stim_start=stim_times[0])
    full_fanos = getFanoFactorFromFiles(h5_file_list, region, window_size)
    mw_res, last_unstimulated_window_time, first_all_stimulated_window_time, last_unstimulated_window_ind, first_all_stimulated_window_ind = runFanoStatTest(full_fanos, h5_file_list[0], region)
    last_unstimulated_window_time, first_all_stimulated_window_time = [last_unstimulated_window_time, first_all_stimulated_window_time] - time_adjustor
    num_cells, num_windows = full_fanos.shape
    mean_full_fanos = full_fanos.mean(axis=0)
    std_err_full_fanos = full_fanos.std(axis=0)/np.sqrt(num_cells)
    sig_indicator_y = np.min((mean_full_fanos - std_err_full_fanos)[[last_unstimulated_window_ind, first_all_stimulated_window_ind]])
    sig_indicator_y = sig_indicator_y - sig_indicator_y*0.031
    sig_indicator_base = sig_indicator_y - sig_indicator_y*0.02
    sig_indicator_text = sig_indicator_base - sig_indicator_base*0.005
    sig_indicator_x = [last_unstimulated_window_time, first_all_stimulated_window_time]
    plt.plot(x_axis, mean_full_fanos, color=colour)
    plt.fill_between(x_axis, mean_full_fanos - std_err_full_fanos, mean_full_fanos + std_err_full_fanos, color=colour, alpha=0.25, label='Std. Err.')
    if mw_res.pvalue < 0.005:
        plt.plot([sig_indicator_x[0], sig_indicator_x[0], sig_indicator_x[1], sig_indicator_x[1]], [sig_indicator_y, sig_indicator_base, sig_indicator_base, sig_indicator_y], lw=1.5, c='k')
        plt.text(np.mean(sig_indicator_x), sig_indicator_text, '***', ha='center', va='top', color='k')
    lower_bound=plt.ylim()[0]
    plotShadedStimulus([0.0], [stim_times[1] - stim_times[0]], plt.ylim()[1], lower_bound=lower_bound) if stim_times != [] else None
    plot_peak = np.max(mean_full_fanos + std_err_full_fanos)
    plt.ylim((lower_bound, plot_peak))
    plt.ylabel('Mean Fano Factor', fontsize='x-large')
    plt.xlabel('Time (s)', fontsize='x-large')
    plt.xlim((x_axis[0], x_axis[-1]))
    plt.legend(fontsize='large')
    plt.title(region.replace('_', ' ').capitalize() + ', n = ' + str(num_cells) + ' cells', fontsize='large') if use_title else None
    plt.tight_layout() if is_tight_layout else None
    return None

def plotRastersForRegions(adj_cell_ids, cell_info, spike_time_dict, regions, stim_info, image_dir):
    """
    For plotting rasters for regions.
    Arguments:  adj_cell_ids, the adjusted cell ids
                spike_time_dict, adj_cell_id => spike times
                regions, str, the list of regions
                stim_info, dataframe
                image_dir, str
    Returns:    file_names
    """
    regions = [regions] if str == type(regions) else regions
    grouped_cell_info = cell_info.loc[adj_cell_ids,['cluster_id','region']].groupby('region')
    trial_info = stim_info.loc[np.random.choice(stim_info[stim_info['stim_ids'] != 17].index)]
    time_info = trial_info - trial_info.stim_starts
    save_names = []
    for region in regions:
        region_fig, region_ax = plt.subplots(nrows=1,ncols=1,figsize=(8,3))
        region_ids = grouped_cell_info.get_group(region).index.values
        for i,id in enumerate(region_ids):
            spike_times = spike_time_dict.get(id)
            relevant_spike_times = spike_times[(trial_info.read_starts < spike_times) & (spike_times < trial_info.read_stops)]
            if len(relevant_spike_times) > 0:
                region_ax.vlines(x=relevant_spike_times - trial_info.stim_starts, ymin=i, ymax=i+1)
        plotShadedStimulus([time_info.stim_starts], [time_info.stim_stops], upper_bound=len(region_ids)+1, lower_bound=0)
        region_ax.set_xlim([time_info.read_starts, time_info.read_stops])
        region_ax.set_ylim([0,len(region_ids)+1])
        region_ax.set_ylabel('Cells', fontsize='x-large')
        region_ax.set_xlabel('Time (s)', fontsize='x-large')
        region_ax.set_yticks([])
        region_ax.tick_params(axis='x', labelsize='large')
        region_ax.set_title(region.capitalize().replace('_',' ') + ' raster', fontsize='x-large')
        plt.tight_layout()
        save_name = os.path.join(image_dir,'Rasters',region,'_'.join([region,'raster',str(int(trial_info.stim_ids))])+'.png')
        os.path.isdir(os.path.dirname(save_name)) or os.makedirs(os.path.dirname(save_name))
        plt.savefig(save_name)
        plt.savefig(save_name.replace('.png','.svg'))
        plt.savefig(save_name.replace('.png','.eps'))
        save_names.append(save_name)
        plt.close('all')
    return save_names

def plotLogLikelihoodsHistograms(log_likes, region, image_dir, bin_width, title='unstimulated', labels=['Binomial', 'Beta-binomial', 'COMb'], colours=['blue','orange','green']):
    """
    For plotting 6 histograms, three for unstimulated, three for stimulated. Want to show if COMb is clearly better than the other two.
    Arguments:  log_likes, numpy array (num_files, 3), binom, betabinom, and comb log likes
                region, string
    Returns:    nothing
    """
    label_to_file_part = {'Binomial':'binom', 'Beta-binomial':'beta_binom', 'COMb':'comb'}
    ll_max, ll_min = log_likes.max(), log_likes.min()
    ll_bins = np.linspace(ll_min, ll_max, 21)
    for i in range(3):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4))
        ax.hist(log_likes[:,i], bins=ll_bins, label=labels[i], color=colours[i])
        ax.legend(fontsize='large')
        ax.set_xlabel('Log Likelihood', fontsize='large')
        ax.set_title(title.capitalize(), fontsize='x-large') if (title != '') & (i == 0) else None
        ax.tick_params(axis='both', labelsize='large')
        plt.tight_layout()
        save_name = os.path.join(image_dir, 'Log_Likelihood', region, str(int(1000*bin_width)) + 'ms', region + '_' + label_to_file_part[labels[i]] + '_' + str(int(1000*bin_width)) + 'ms' + '_' + title + '_log_likelihood_hist.png')
        os.makedirs(os.path.dirname(save_name)) if not os.path.exists(os.path.dirname(save_name)) else None
        plt.savefig(save_name)
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Saved: ' + save_name)
        plt.close('all')

def plotParaHistogram(image_dir, param, region, bin_width, stim, param_values, x_label='', title=''):
    """
    For plotting parameter histograms.
    Arguments:  image_dir, str
                param, the param name
                region,
                bin_width,
                stim, str ['stim' or 'unstim']
                param_values, array
                x_label,
                title,
    Returns:    nothing
    """
    fig, ax = plt.subplots(figsize=(5,4))
    ax.hist(param_values)
    ax.set_xlabel(x_label, fontsize='x-large') if x_label != '' else None
    ax.set_title(title, fontsize='large') if title != '' else None
    ax.tick_params(axis='both', labelsize='large')
    plt.tight_layout()
    save_name = os.path.join(image_dir, 'Param_hists', region, str(int(1000*bin_width)) + 'ms', '_'.join([region, str(int(1000*bin_width)) + 'ms', param, stim]) + '.png')
    os.makedirs(os.path.dirname(save_name)) if not os.path.exists(os.path.dirname(save_name)) else None
    plt.savefig(save_name)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Saved: ' + save_name)
    plt.close('all')
    
def plotTwoParamHistogram(image_dir, distn, x_param, y_param, region, bin_width, stim, x_param_values, y_param_values, title=''):
    """
    For plotting 2d param histograms for betabinomial and COMb distributions.
    Arguments:  image_dir, str
                distn, str, the distribution, for saving purposes
                x_param, the x-params label
                y_param, the y_param label
                region, str
                bin_width,
                stim, str ['stim' or 'unstim']
                x_param_values, floats
                y_param_values, floats
                title,
    Returns:    nothing
    """
    fig, ax = plt.subplots(figsize=(5,4))
    h, x_edge, y_edge, cs = ax.hist2d(x=x_param_values, y=y_param_values, bins=10, cmap=cm.PuBu_r)
    ax.set_xlabel(x_param, fontsize='x-large')
    ax.set_ylabel(y_param, fontsize='x-large')
    ax.tick_params(axis='both', labelsize='large')
    ax.set_title(title) if title !='' else None
    cbar = fig.colorbar(cs)
    plt.tight_layout()
    save_name = os.path.join(image_dir, 'Param_hists', region, str(int(1000*bin_width)) + 'ms', '_'.join([region, str(int(1000*bin_width)) + 'ms', distn, stim]) + '.png')
    os.makedirs(os.path.dirname(save_name)) if not os.path.exists(os.path.dirname(save_name)) else None
    plt.savefig(save_name)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Saved: ' + save_name)
    plt.close('all')

