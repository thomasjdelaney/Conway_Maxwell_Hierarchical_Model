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
from itertools import product

parser = argparse.ArgumentParser(description='For loading in the functions and loading the cell info.')
parser.add_argument('-b', '--bin_width', help='Time bin with to use (in seconds).', default=0.001, type=float)
parser.add_argument('-r', '--region', help='The region to use for any ad hoc plotting.', default='thalamus', type=str)
parser.add_argument('-w', '--window_size', help='The number of bins to use for fitting.', default=100, type=int)
parser.add_argument('-t', '--plot_trial_summaries', help='Flag to plot the trial summaries, or skip them.', default=False, action='store_true')
parser.add_argument('-a', '--plot_averages', help='Flag to plot the averages across trials', default=False, action='store_true')
parser.add_argument('-f', '--plot_fano', help='Flag to plot the fano factors.', default=False, action='store_true')
parser.add_argument('-c', '--compare_dists', help='Flag to plot the distribution comparison plot.', default=False, action='store_true')
parser.add_argument('-d', '--debug', help='Enter debug mode.', default=False, action='store_true')
args = parser.parse_args()

pd.set_option('max_rows', 30)
np.set_printoptions(linewidth=shutil.get_terminal_size().columns)

proj_dir = os.path.join(os.environ['PROJ'], 'Conway_Maxwell_Hierarchical_Model')
csv_dir = os.path.join(proj_dir, 'csv')
mat_dir = os.path.join(proj_dir, 'mat')
py_dir = os.path.join(proj_dir, 'py')
h5_dir = os.path.join(proj_dir, 'h5')
image_dir = os.path.join(proj_dir, 'images')
posterior_dir = os.path.join(proj_dir, 'posterior')
frontal_dir = os.path.join(proj_dir, 'frontal')
fig_dir = os.path.join(proj_dir, 'latex', 'figures')

sys.path.append(py_dir)
sys.path.append(os.path.join(os.environ['PROJ'], 'Conway_Maxwell_Binomial_Distribution'))
import ConwayMaxwellHierarchicalModel as comh
import ConwayMaxwellBinomial as comb

def plotAveragesAcrossTrials(h5_file_list, title, file_name_suffix, stim_times, measure_list, y_label_list, index_list, label_list, reparametrise_list, file_name_prefix_list):
    """
    For plotting the average of measures across trials. Loops through all those lists.
    Arguments:  h5_file_list, list of str
                title, str,
                file_name_suffix, str, for every file
                stim_times, start and end time of stimuli
                measure_list, list of str
                index_list, index into the measures or not
                label_list, str, labels,
                reparametrise_list, list of booleans
                file_name_prefix_list, list of str
    Returns:    Nothing
    """
    for measure, y_label, index, label, reparametrise, file_name_prefix in zip(measure_list,y_label_list,index_list,label_list,reparametrise_list,file_name_prefix_list):
        plt.figure(figsize=(9,3))
        comh.plotAverageMeasure(h5_file_list, args.region, measure, index=index, stim_times=stim_times, label=label, title=title, reparametrise=reparametrise, y_label=y_label)
        save_dir = os.path.join(image_dir, 'Averaging_measurements_across_trials', args.region, str(int(1000*args.bin_width)) + 'ms', measure)
        os.makedirs(save_dir) if not os.path.exists(save_dir) else None
        save_name = os.path.join(save_dir, file_name_prefix + file_name_suffix)
        plt.savefig(save_name); plt.savefig(save_name.replace('.png','.svg'));
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Saved: ' + save_name)
        plt.close('all')
    return None

def plotDistributions(distributions, labels, colours, n, use_legend=False):
    """
    For plotting comparisons of COMb and binomial and beta-binomial distributions.
    Arguments:  distributions, list of distributions
                labels, list of str
                colours, list of str
                n, int, number of neurons/bernoulli trials
    Returns:    nothing
    """
    x_range = range(n+1)
    x_lims=[0,n]
    for i, (dist, colour, label) in enumerate(zip(distributions, colours, labels)):
        plt.plot(x_range, dist.pmf(x_range), color=colour, label=label)
    plt.ylabel('P(k)', fontsize='x-large')
    plt.xlabel('k', fontsize='x-large')
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.legend(fontsize='large') if use_legend else None
    plt.xlim(x_lims)
    plt.tight_layout()
    return None

def plotBinCombKLDiv(n):
    """
    For plotting the heatmap of the KL divergence between a binomial distribution and a COMb distribution as a function of p and nu
    Arguments:  n, int
    Returns:    nothing
    """
    possible_p_values = np.linspace(0.0001,0.9999, 101)
    possible_nu_values = np.linspace(0.5, 2.0, 101)
    grid_p, grid_nu = np.meshgrid(possible_p_values, possible_nu_values)
    d_kl_values = np.zeros(grid_p.shape)
    for i,j in product(range(grid_p.shape[0]), range(grid_p.shape[1])):
        d_kl_values[i,j] = comh.binCombDkl(n, grid_p[i,j], grid_nu[i,j])
    plt.contourf(grid_p, grid_nu, d_kl_values, levels=100)
    plt.xlabel('$p$', fontsize='x-large')
    plt.ylabel(r'$\nu$', fontsize='x-large')
    plt.colorbar()
    plt.title(r'$D_{KL}(COMb, Binomial)$', fontsize='x-large')
    plt.tight_layout()
    return None

if not args.debug:
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Starting main function...')
    stim_info, stim_ids = comh.loadStimulusInfo(mat_dir)

    ##### Average across all stimulated trials ##################
    if args.plot_averages:
        h5_file_list = comh.getFileListFromTrialIndices(h5_dir, stim_info[stim_info['stim_ids'] != 17].index.values, args.bin_width, args.window_size)
        title = args.region.capitalize().replace('_', ' ') + ', Num Trials=' + str(len(h5_file_list)) + ' All stimulated trials'
        file_name_suffix = '_all_stimulated_trials.png'
        trial_info = stim_info.loc[comh.getTrialIndexFromH5File(h5py.File(h5_file_list[0],'r'))]
        stim_times = [trial_info['stim_starts'], trial_info['stim_stops']]
        measure_list = ['moving_avg', 'moving_var', 'corr_avg', 'binom_params', 'comb_params', 'comb_params', 'betabinom_ab', 'betabinom_ab']
        y_label_list = ['Act. Cells Moving Avg.', 'Act. Cells Moving Var.', 'Avg. Corr. Coef.', r'Bin. $p$', r'COMb $p$', r'COMb $\nu$', r'Beta-bin $\pi$', r'Beta-bin $\rho$']
        index_list = [None, None, None, None, 0, 1, 0, 1]
        label_list = ['Moving Avg.', 'Moving Var.', 'Avg. Corr.', r'Binomial $p$', r'COM-Binomial $p$', r'COM-Binomial $\nu$', r'Beta-Binomial $\pi$', r'Beta-Binomial $\rho$']
        reparametrise_list = [False, False, False, False, False, False, True, True]
        file_name_prefix_list = ['moving_avg', 'moving_var', 'corr_avg', 'binom_p', 'comb_p', 'comb_nu', 'betabinom_pi', 'betabinom_rho']
        plotAveragesAcrossTrials(h5_file_list, title, file_name_suffix, stim_times, measure_list, y_label_list, index_list, label_list, reparametrise_list, file_name_prefix_list)

    ######## Average across all unstimulated trials ##########
        h5_file_list = comh.getFileListFromTrialIndices(h5_dir, stim_info[stim_info['stim_ids'] == 17].index.values, args.bin_width, args.window_size)
        title = args.region.capitalize().replace('_', ' ') + ', Num Trials=' + str(len(h5_file_list)) + ' All unstimulated trials'
        file_name_suffix = '_all_unstimulated_trials.png'
        trial_info = stim_info.loc[comh.getTrialIndexFromH5File(h5py.File(h5_file_list[0],'r'))]
        stim_times = [trial_info['stim_starts'], trial_info['stim_stops']]
        plotAveragesAcrossTrials(h5_file_list, title, file_name_suffix, stim_times, measure_list, y_label_list, index_list, label_list, reparametrise_list, file_name_prefix_list)

    ################# Plotting Trial summaries ###################
    if args.plot_trial_summaries:
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Plotting trial summaries...')
        h5_file_list = comh.getFileListFromTrialIndices(h5_dir, list(range(170)), args.bin_width, args.window_size)
        save_dir = os.path.join(image_dir, 'Trial_summaries', args.region, str(int(1000*args.bin_width)) + 'ms')
        os.makedirs(save_dir) if not os.path.exists(save_dir) else None
        for h5_file_name in h5_file_list:
            h5_file = h5py.File(h5_file_name, 'r')
            comh.plotTrialSummary(h5_file, args.region, stim_info)
            save_name = os.path.join(save_dir, 'trial_summary_' + str(comh.getTrialIndexFromH5File(h5_file)) + '.png')
            plt.savefig(save_name);plt.savefig(save_name.replace('.png', '.svg'));
            plt.close('all')
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Trial summaries complete: ' + save_dir)

    ################## Fano Factors ###############################
    if args.plot_fano:
        h5_file_list = comh.getFileListFromTrialIndices(h5_dir, stim_info[stim_info['stim_ids'] != 17].index.values, args.bin_width, args.window_size)
        trial_info = stim_info.loc[comh.getTrialIndexFromH5File(h5py.File(h5_file_list[0],'r'))]
        stim_times = [trial_info['stim_starts'], trial_info['stim_stops']]
        plt.figure(figsize=(5.5,4))
        comh.plotCellFanoFactors(h5_file_list, args.region, stim_times=stim_times, colour='blue', is_tight_layout=True, use_title=True, window_size=args.window_size)
        save_name = os.path.join(image_dir, 'Fano_factors', args.region, str(int(1000*args.bin_width)) + 'ms', args.region + '_' + str(int(1000*args.bin_width)) + 'ms' + '_fano_factor.png')
        os.makedirs(os.path.dirname(save_name)) if not os.path.exists(os.path.dirname(save_name)) else None
        plt.savefig(save_name)
        save_name = os.path.join(image_dir, 'Fano_factors', args.region, str(int(1000*args.bin_width)) + 'ms', args.region + '_' + str(int(1000*args.bin_width)) + 'ms' + '_fano_factor.svg')
        plt.savefig(save_name)
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Fano factor plots saved: ' + save_name)


    ################ Comparing distributions #####################
    if args.compare_dists:
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Plotting distribution comparisons...')
        [n,p,a,b]=100,0.5,10,10
        plt.figure(figsize=(4,3))
        plotDistributions([binom(n,p), betabinom(n,a,b)], ['Binom PMF', 'Beta-binom PMF'], ['blue', 'orange'], n, use_legend=True)
        y_lims = plt.ylim()
        save_name = os.path.join(fig_dir, 'betabinomial_overdispersion.png')
        plt.savefig(save_name); plt.savefig(save_name.replace('.png','.svg')); plt.savefig(save_name.replace('.png','.eps'))
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Beta-binom overdispersion: ' + save_name)
        plt.close('all')
        [a,b] = [0.15, 0.15]
        plt.figure(figsize=(4,3))
        plotDistributions([binom(n,p), betabinom(n,a,b)], ['Binom PMF', 'Beta-binom PMF'], ['blue', 'orange'], n)
        plt.ylim(y_lims)
        save_name = os.path.join(fig_dir, 'betabinomial_big_overdispersion.png')
        plt.savefig(save_name); plt.savefig(save_name.replace('.png','.svg')); plt.savefig(save_name.replace('.png','.eps'))
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Beta-binom big overdispersion: ' + save_name)
        plt.close('all')
        a,b,nu=10,10,2.5
        plt.figure(figsize=(4,3))
        plotDistributions([binom(n,p), betabinom(n,a,b), comb.ConwayMaxwellBinomial(p,nu,n)], ['Binom PMF', 'Beta-binom PMF', 'COMb PMF'], ['blue', 'orange', 'green'], n, use_legend=True)
        save_name = os.path.join(fig_dir, 'comb_underdispersion.png')
        plt.savefig(save_name); plt.savefig(save_name.replace('.png','.svg')); plt.savefig(save_name.replace('.png','.eps'))
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'COMb under-dispersion: ' + save_name)
        plt.close('all')
        plt.figure(figsize=(4,3))
        plotBinCombKLDiv(n)
        save_name = os.path.join(fig_dir, 'comb_bin_dkl.png')
        plt.savefig(save_name); plt.savefig(save_name.replace('.png','.svg')); plt.savefig(save_name.replace('.png','.eps'))
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'COMb-bin Dkl: ' + save_name)
        plt.close('all')
        a,b,nu=10,10,0.1
        plt.figure(figsize=(4,3))
        plotDistributions([binom(n,p), betabinom(n,a,b), comb.ConwayMaxwellBinomial(p,nu,n)], ['Binom PMF', 'Beta-binom PMF', 'COMb PMF'], ['blue', 'orange', 'green'], n)
        save_name = os.path.join(fig_dir, 'comb_overrdispersion.png')
        plt.savefig(save_name); plt.savefig(save_name.replace('.png','.svg')); plt.savefig(save_name.replace('.png','.eps'))
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'COMb over-dispersion: ' + save_name)
        plt.close('all')
        p=0.4; a=10; b = (a/p)*(1-p);
        plt.figure(figsize=(4,3))
        plotDistributions([binom(n,p), betabinom(n,a,b), comb.ConwayMaxwellBinomial(p,nu,n)], ['Binom PMF', 'Beta-binom PMF', 'COMb PMF'], ['blue', 'orange', 'green'], n)
        save_name = os.path.join(fig_dir, 'comb_skewed.png')
        plt.savefig(save_name); plt.savefig(save_name.replace('.png','.svg')); plt.savefig(save_name.replace('.png','.eps'))
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'COMb skewed: ' + save_name)
        plt.close('all')
