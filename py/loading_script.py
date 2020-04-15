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

parser = argparse.ArgumentParser(description='For loading in the functions and loading the cell info.')
parser.add_argument('-n', '--num_cells', help='Number of cells to use.', default=100, type=int)
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

def easyCombLogLikeFit(data, init, bounds, n):
    """
    For fitting a Conway-Maxwell-binomial distribution using the logpmf function.
    Arguments:  data,
                init, initial guess,
                bounds,
                n, the number of cells
    Returns:    fitted distribution object
    """
    res = minimize(lambda x:comb.ConwayMaxwellBinomial(x[0], x[1], n).logpmf(data).sum(), init, bounds=bounds)
    fitted = comb.ConwayMaxwellBinomial(m=n, p=res.x[0], nu=res.x[1])
    return fitted

if not args.debug:
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Starting main function...')
    cell_info = comh.loadCellInfo(csv_dir)
    stim_info, stim_ids = comh.loadStimulusInfo(mat_dir)
    adj_cell_ids = comh.getRandomSubsetOfCells(cell_info, args.num_cells)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading spike time dictionary...')
    spike_time_dict = comh.loadSpikeTimeDict(adj_cell_ids, posterior_dir, frontal_dir, cell_info)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loaded.')
    interval_start_time = stim_info.loc[0]['stim_starts'] - 0.5
    interval_end_time = stim_info.loc[2]['stim_stops'] + 0.5
    bin_width = 0.001
    region_to_spike_time_dict = comh.divideSpikeTimeDictByRegion(spike_time_dict,cell_info)
    bin_borders, region_to_active_cells = comh.getNumberOfActiveCellsByRegion(interval_start_time, interval_end_time, bin_width, region_to_spike_time_dict)
    plt.figure(figsize=(5,4))
    comh.plotNumActiveCellsByTimeByRegion(bin_borders, region_to_active_cells, stim_starts=stim_info.stim_starts[:3].values, stim_stops=stim_info.stim_stops[:3].values)
    num_active_cells_binned = region_to_active_cells.get('thalamus')
    total_cells = len(region_to_spike_time_dict.get('thalamus'))
    fitted_binom = comh.fitBinomialDistn(num_active_cells_binned, total_cells)
    plt.figure(figsize=(5,4))
    comh.plotCompareDataFittedDistn(num_active_cells_binned, fitted_binom)
    fitted_betabinom = easyLogLikeFit(betabinom, num_active_cells_binned, [1.0, 1.0], [(np.finfo(float).resolution, None), (np.finfo(float).resolution, None)], total_cells)
    plt.figure(figsize=(5,4))
    comh.plotCompareDataFittedDistn(num_active_cells_binned, fitted_betabinom)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Fitting Conway-Maxwell-binomial distribution...')
    fitted_comb = easyCombLogLikeFit(num_active_cells_binned, [0.5, 0], [(0.00001, 1 - 0.00001), (-2,2)], total_cells)
    # fitted_comb_params = comb.estimateParams(total_cells, num_active_cells_binned, [0.5, 0])
    # fitted_comb = comb.ConwayMaxwellBinomial(fitted_comb_params[0], fitted_comb_params[1], total_cells)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Fitted.')
    plt.figure(figsize=(5,4))
    comh.plotCompareDataFittedDistn(num_active_cells_binned, fitted_comb)

    # plt.show(block=False)

# TODO Beta binomial distribution, multiplicative binomial distribution, conway maxwell binomial distribution
