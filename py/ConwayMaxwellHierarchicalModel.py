import os, sys, h5py
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from scipy.io import loadmat
from multiprocessing import Pool
from scipy.stats import binom, betabinom
from scipy.optimize import minimize

sys.path.append(os.path.join(os.environ['PROJ'], 'Conway_Maxwell_Binomial_Distribution'))
import ConwayMaxwellBinomial as comb

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
        'stim_stops':stim_info_dict['stimStops'][0]})
    stim_info['read_starts'] = stim_info['stim_starts'] - getExtraTrialTime(stim_info)
    stim_info['read_stops'] = stim_info['stim_stops'] + getExtraTrialTime(stim_info)
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
    return spike_counts > 0

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
    return bin_borders, np.vstack(is_active_future.get()).sum(axis=0)

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
    for region,regional_spike_time_dict in region_to_spike_time_dict.items():
        bin_borders, num_active_cells_binned = getNumberOfActiveCellsInBinnedInterval(interval_start_time, interval_end_time, bin_width, regional_spike_time_dict)
        region_to_active_cells[region] = num_active_cells_binned
    return bin_borders, region_to_active_cells

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

def getTrialMeasurements(num_active_cells_binned, is_stimulated, window_inds, num_cells, window_size=100, window_skip=10):
    """
    Get all the measurements we want for a trial, providing start and stop times, and id.
    Arguments:  num_active_cells_binned, numpy array (int)
                is_stimulated, numpy array boolean, whether or not the stimulus was present for the entirety of each bin.
                window_inds, numpy array (int) (num_windows, window_size), indices for each window, each row is a list of indices for a window.
                num_cells, int
                window_size, int, the number of bins we will use to fit the distributions
                window_skip, int, the number of bins between each fitting window. This influences how many times we have to fit each distribution,
                    and therefore has a major effect on how long it takes to process everything.
    Returns:    moving_avg, 
                binom_params, 
                betabinom_params, 
                comb_params,
                binom_log_like,
                betabinom_log_like,
                comb_log_like.
    """
    num_windows = window_inds.shape[0]
    windowed_counts = num_active_cells_binned[window_inds]
    moving_avg = windowed_counts.mean(axis=1)
    all_stimulated = is_stimulated[window_inds].all(axis=1)
    any_stimulated = is_stimulated[window_inds].any(axis=1)
    with Pool() as pool:
        binom_params_future = pool.starmap_async(fitBinomialDistn, zip(windowed_counts, [num_cells]*num_windows))
        betabinom_params_future = pool.starmap_async(easyLogLikeFit, zip([betabinom]*num_windows, windowed_counts, [[1.0,1.0]]*num_windows, [((1e-08,None),(1e-08,None))]*num_windows, [num_cells]*num_windows))
        comb_params_future = pool.starmap_async(comb.estimateParams, zip([num_cells]*num_windows, windowed_counts))
        binom_params_future.wait()
        betabinom_params_future.wait()
        comb_params_future.wait()
    binom_params = np.array([b.args[1] for b in binom_params_future.get()])
    betabinom_ab = np.array([[b.kwds['a'],b.kwds['b']] for b in betabinom_params_future.get()])
    comb_params = np.array(comb_params_future.get())
    binom_log_like = np.array([binom.logpmf(fc, num_cells, p).sum() for fc,p in zip(windowed_counts, binom_params)])
    betabinom_log_like = np.array([betabinom.logpmf(fc, num_cells, p[0], p[1]).sum() for fc,p in zip(windowed_counts, betabinom_ab)])
    comb_log_like = -np.array([comb.conwayMaxwellNegLogLike(p, num_cells, fc) for fc,p in zip(windowed_counts, comb_params)])
    return moving_avg, all_stimulated, any_stimulated, binom_params, binom_log_like, betabinom_ab, betabinom_log_like, comb_params, comb_log_like

def isStimulatedBins(bin_borders, stim_start, stim_stop):
    """
    Get an array of booleans indicating which bins are stimulated and which are not.
    Arguments:  bin_borders, the times where the bins start and stop in seconds
                stim_start, time,
                stim_stop, time
    Returns:    numpy array boolean
    """
    return np.array([(bin_start > stim_start) & (bin_stop < stim_stop) for bin_start,bin_stop in zip(bin_borders[:-1],bin_borders[1:])])

def getH5FileName(h5_dir, trial_index, bin_width, num_bins_fitting):
    """
    For getting the name of the h5 file. Can be used for saving or loading or whatever.
    Arguments:  h5_dir, string, the directory
                trial_index, int, the index of the trial 
                bin_width, float,
                num_bins_fitting, number of bins used to fit the distributions.
    Returns:    string
    """
    return os.path.join(h5_dir, 'trial_' + str(trial_index) + '_bin_width_' + str(int(1000*bin_width)) + 'ms_num_bins_' + str(num_bins_fitting) + '.h5')

def getBinCentres(bin_borders):
    """
    Function for getting the centre of bins given the borders.
    Arguments:  bin_borders, the borders of each bin
    Returns:    bin_centres, length(bin_borders) - 1, the centre of each time bin.
    """
    return (bin_borders[:-1] + bin_borders[1:])/2

##########################################################
########## PLOTTING FUNCTIONS ############################
##########################################################

def plotShadedStimulus(stim_starts, stim_stops, upper_bound):
    """
    For plotting a shaded aread to represent the times when a stimulus was present.
    Arguments:  stim_starts, list or array, the start times of the stimuli
                stim_stops, list or array, the stop times of the stimuli
    Returns:    nothing
    """
    for i,(stim_start, stim_stop) in enumerate(zip(stim_starts, stim_stops)):
        plt.fill_between(x=[stim_start, stim_stop], y1=upper_bound, y2=0, color='grey', alpha=0.3)
    return None

def plotNumActiveCellsByTime(bin_borders, num_active_cells_binned, stim_starts=[], stim_stops=[], **kwargs):
    """
    For plotting the number of active cells across bins, optionally with stimulus time shaded.
    Arguments:  bin_borders, array (float) 
                num_active_cells_binned, array (int)
                stim_starts, list or array, the start times of stimuli
                stim_stops, list or array, the stop times of stimuli
    Returns:    Nothing
    """
    bin_centres = getBinCentres(bin_borders)
    if (len(stim_starts) > 0) & (len(stim_stops) > 0):
        plotShadedStimulus(stim_starts, stim_stops, num_active_cells_binned.max())
    plt.plot(bin_centres, num_active_cells_binned, **kwargs)
    return None

def plotNumActiveCellsByTimeByRegion(bin_borders, region_to_active_cells, stim_starts=[], stim_stops=[], is_tight_layout=True, **kwargs):
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
        plotNumActiveCellsByTime(bin_borders, num_active_cells_binned, label=region.capitalize())
    plt.legend(fontsize='large')
    plt.xlabel('Time (s)', fontsize='large')
    plt.ylabel('Num. Active Cells', fontsize='large')
    plt.tight_layout() if is_tight_layout else None
    return None

def plotCompareDataFittedDistn(num_active_cells_binned, fitted_distn, plot_type='pdf', data_label='Num. Active Cells', distn_label=['Fitted Distn. PMF'], title=''):
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
    plt.hist(num_active_cells_binned, bins=bin_borders, density=True, label=data_label, align='left')
    for distn, d_label in zip(fitted_distn,distn_label):
        plt.plot(bin_borders, distn.pmf(bin_borders), label=d_label)
    plt.legend(fontsize='large')
    plt.xlabel('Num. Active Cells', fontsize='large')
    plt.ylabel('P(k)', fontsize='large')
    plt.title(title, fontsize='large') if title != '' else None
    plt.tight_layout()
    return None
