import os
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from scipy.io import loadmat
from multiprocessing import Pool
from scipy.stats import nbinom, binom, betabinom

def loadCellInfo(csv_dir):
    """
    For loading the csv containing information about each cell. 
    Arguments:  csv_dir, the directory where the file can be found
    Returns:    pandas DataFrame
    """
    return pd.read_csv(os.path.join(csv_dir, 'cell_info.csv'), index_col='adj_cluster_id')

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
        print(dt.datetime.now().isoformat() + 'WARN: ' + 'Returning all available cells.')
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
    for region,spike_time_dict in region_to_spike_time_dict.items():
        bin_borders, num_active_cells_binned = getNumberOfActiveCellsInBinnedInterval(interval_start_time, interval_end_time, bin_width, spike_time_dict)
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
    bin_centres = (bin_borders[:-1] + bin_borders[1:])/2
    if (len(stim_starts) > 0) & (len(stim_stops) > 0):
        plotShadedStimulus(stim_starts, stim_stops, num_active_cells_binned.max())
    plt.plot(bin_centres, num_active_cells_binned, **kwargs)
    return None

def plotNumActiveCellsByTimeByRegion(bin_borders, region_to_active_cells, stim_starts=[], stim_stops=[], **kwargs):
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
    plt.tight_layout()
    return None

def plotCompareDataFittedDistn(num_active_cells_binned, fitted_distn, plot_type='pdf', data_label='Num. Active Cells', distn_label='Fitted Distn. PMF'):
    """
    For comparing a fitted distribution to some data. (pdf and/or cdf?)
    Arguments:  num_active_cells_binned, array int
                fitted_distn, distribution object, needs pdf and cdf functions
                plot_type, ['pdf', 'cdf']
    Returns:    nothing
    """
    bin_borders = range(num_active_cells_binned.max()+1)
    plt.hist(num_active_cells_binned, bins=bin_borders, density=True, label=data_label, align='left')
    plt.plot(bin_borders, fitted_distn.pmf(bin_borders), label=distn_label)
    plt.legend(fontsize='large')
    plt.xlabel('Num. Active Cells', fontsize='large')
    plt.ylabel('P(k)', fontsize='large')
    plt.tight_layout()
    return None
