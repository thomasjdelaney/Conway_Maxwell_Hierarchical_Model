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
parser.add_argument('-b', '--bin_width', help='Time bin with to use (in seconds).', default=0.001, type=float)
parser.add_argument('-r', '--region', help='The region to use for any ad hoc plotting.', default='thalamus', type=str)
parser.add_argument('-w', '--window_size', help='The number of bins to use for fitting.', default=100, type=int)
parser.add_argument('-t', '--plot_trial_summaries', help='Flag to plot the trial summaries, or skip them.', default=False, action='store_true')
parser.add_argument('-a', '--plot_averages', help='Flag to plot the averages across trials', default=False, action='store_true')
parser.add_argument('-f', '--plot_fano', help='Flag to plot the fano factors.', default=False, action='store_true')
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

sys.path.append(py_dir)
sys.path.append(os.path.join(os.environ['PROJ'], 'Conway_Maxwell_Binomial_Distribution'))
import ConwayMaxwellHierarchicalModel as comh
import ConwayMaxwellBinomial as comb

def plotAveragesAcrossTrials(h5_file_list, title, file_name_suffix, stim_times, measure_list, index_list, label_list, reparametrise_list, file_name_prefix_list):
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
    for measure, index, label, reparametrise, file_name_prefix in zip(measure_list,index_list,label_list,reparametrise_list,file_name_prefix_list):
        plt.figure(figsize=(10,4))
        comh.plotAverageMeasure(h5_file_list, args.region, measure, index=index, stim_times=stim_times, label=label, title=title, reparametrise=reparametrise)
        save_dir = os.path.join(image_dir, 'Averaging_measurements_across_trials', args.region, str(int(1000*args.bin_width)) + 'ms', measure)
        os.makedirs(save_dir) if not os.path.exists(save_dir) else None
        save_name = os.path.join(save_dir, file_name_prefix + file_name_suffix)
        plt.savefig(save_name); plt.savefig(save_name.replace('.png','.svg'))
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Saved: ' + save_name)
        plt.close('all')
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
        index_list = [None, None, None, None, 0, 1, 0, 1]
        label_list = ['Moving Avg.', 'Moving Var.', 'Avg. Corr.', r'Binomial $p$', r'COM-Binomial $p$', r'COM-Binomial $\nu$', r'Beta-Binomial $\pi$', r'Beta-Binomial $\rho$']
        reparametrise_list = [False, False, False, False, False, True, True]
        file_name_prefix_list = ['moving_avg', 'moving_var', 'corr_avg', 'binom_p', 'comb_p', 'comb_nu', 'betabinom_pi', 'betabinom_rho']
        plotAveragesAcrossTrials(h5_file_list, title, file_name_suffix, stim_times, measure_list, index_list, label_list, reparametrise_list, file_name_prefix_list)

    ######## Average across all unstimulated trials ##########
        h5_file_list = comh.getFileListFromTrialIndices(h5_dir, stim_info[stim_info['stim_ids'] == 17].index.values, args.bin_width, args.window_size)
        title = args.region.capitalize().replace('_', ' ') + ', Num Trials=' + str(len(h5_file_list)) + ' All unstimulated trials'
        file_name_suffix = '_all_unstimulated_trials.png'
        trial_info = stim_info.loc[comh.getTrialIndexFromH5File(h5py.File(h5_file_list[0],'r'))]
        stim_times = [trial_info['stim_starts'], trial_info['stim_stops']]
        plotAveragesAcrossTrials(h5_file_list, title, file_name_suffix, stim_times, measure_list, index_list, label_list, reparametrise_list, file_name_prefix_list) 

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
