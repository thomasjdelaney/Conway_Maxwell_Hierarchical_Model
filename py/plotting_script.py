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
parser.add_argument('-e', '--fitted_example', help='Flag to plot the example of data and fitted distributions.', default=False, action='store_true')
parser.add_argument('-s', '--plot_rasters', help='Flag to plot some rasters for selected regions.', default=False, action='store_true')
parser.add_argument('-l', '--plot_ll_hists', help='Flag for plotting the histograms of log likelihoods', default=False, action='store_true')
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
        plt.figure(figsize=(8,3.1))
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

def findBestDists(h5_file_list, region):
    """
    For finding the the index of the best fitting distribution for every window and every trial
    Arguments:  h5_file_list, list str
                region, str
    Returns:    best_fit_inds, numpy array (int) (num trials, num windows),
    """
    best_fit_inds = np.zeros((len(h5_file_list), h5py.File(h5_file_list[0],'r').get('window_centre_times').size), dtype=int)
    for i,h5_file_name in enumerate(h5_file_list):
        h5_file = h5py.File(h5_file_name,'r')
        binom_log_like = h5_file.get(region).get('binom_log_like')[()]
        betabinom_log_like = h5_file.get(region).get('betabinom_log_like')[()]
        comb_log_like = h5_file.get(region).get('comb_log_like')[()]
        best_fit_inds_file = np.vstack([binom_log_like, betabinom_log_like, comb_log_like]).argmax(axis=0)
        best_fit_inds[i] = best_fit_inds_file
    return best_fit_inds

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
        y_label_list = ['Num. Act. Cells Moving Avg.', 'Num. Act. Cells Moving Var.', 'Avg. Corr. Coef.', r'Bin. $p$', r'COMb $p$', r'COMb $\nu$', r'Beta-bin $\pi$', r'Beta-bin $\rho$']
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
        plt.figure(figsize=(4,3))
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

    ################## Example of fitting #######################
    if (args.fitted_example) & (args.region == 'v1') & (args.bin_width == 0.001):
        h5_file_name = comh.getFileListFromTrialIndices(h5_dir, [0], args.bin_width, args.window_size)[0]
        h5_file = h5py.File(h5_file_name,'r')
        num_cells = h5_file.get(args.region).get('num_cells')[()]
        num_active_cells_binned = h5_file.get(args.region).get('num_active_cells_binned')[2100 + np.arange(h5_file.get('window_size')[()])]
        bin_dist = comh.fitBinomialDistn(num_active_cells_binned, num_cells)
        betabin_dist = comh.easyLogLikeFit(betabinom, num_active_cells_binned, [1.0,1.0], ((1e-08,None),(1e-08,None)), num_cells)
        comb_fitted_params = comb.estimateParams(num_cells, num_active_cells_binned)
        comb_dist = comb.ConwayMaxwellBinomial(comb_fitted_params[0], comb_fitted_params[1], num_cells)
        plt.figure(figsize=(4,3))
        comh.plotCompareDataFittedDistn(num_active_cells_binned, [bin_dist, betabin_dist, comb_dist], plot_type='pdf', data_label='Empirical Distn', distn_label=['Binomial PMF', 'Beta-binomial PMF', 'COM-binomial PMF'], title='', colours=['blue', 'orange', 'green'])
        save_name = os.path.join(fig_dir, 'fitting_example.png')
        plt.savefig(save_name); plt.savefig(save_name.replace('.png','.svg'));
        plt.close('all')

    if args.fitted_example:
        h5_file_list = comh.getFileListFromTrialIndices(h5_dir, range(170), args.bin_width, args.window_size)
        best_fit_inds = findBestDists(h5_file_list, args.region)
        plt.figure(figsize=(4,3))
        n, bins, patches = plt.hist(best_fit_inds.flatten(), bins=[-0.5,0.5,1.5,2.5], density=True)
        csv_file_name = os.path.join(csv_dir, 'best_fits.csv')
        csv_exists = os.path.isfile(csv_file_name)
        if csv_exists:
            prop_frame = pd.read_csv(csv_file_name, index_col=['bin_width','region'])
            prop_frame.loc[(str(int(args.bin_width*1000))+'ms',args.region),:] = n
            os.remove(csv_file_name)
        else:
            prop_frame = pd.DataFrame({'bin_width':[str(int(args.bin_width*1000))+'ms'], 'region':args.region, 'binom_prop':n[0], 'betabinom_prop':n[1], 'comb_binom':n[2]})
            prop_frame = prop_frame.set_index(['bin_width','region'])
        prop_frame.to_csv(csv_file_name, mode='w', header=True, index_label=['bin_width','region'])
        plt.xticks([0,1,2],['Bin.', 'Beta-bin.', 'COMb'], fontsize='large') # 93% COMB
        plt.ylabel('Proportion best fit', fontsize='x-large')
        plt.tight_layout()
        save_name = os.path.join(image_dir, 'Best_fit_proportions', args.region, str(int(1000*args.bin_width)) + 'ms', args.region + '_' + str(int(1000*args.bin_width)) + 'ms' + '_best_fit_proportion.png')
        os.makedirs(os.path.dirname(save_name)) if not os.path.exists(os.path.dirname(save_name)) else None
        plt.savefig(save_name); plt.savefig(save_name.replace('.png','.svg'));
        plt.close('all')

######################## PLOT RASTERS #################################
    if args.plot_rasters:
        cell_info = comh.loadCellInfo(csv_dir)
        adj_cell_ids = comh.getRandomSubsetOfCells(cell_info, 100, groups=['good'], regions=['v1','thalamus','hippocampus'])
        spike_time_dict = comh.loadSpikeTimeDict(adj_cell_ids, posterior_dir, frontal_dir, cell_info)
        save_names = comh.plotRastersForRegions(adj_cell_ids, cell_info, spike_time_dict, ['v1','thalamus','hippocampus'], stim_info, image_dir)
        [print(dt.datetime.now().isoformat() + ' INFO: ' + sn + ' saved.') for sn in save_names]


####################### PLOT LL HISTOGRAMS ###########################
    if args.plot_ll_hists:
        stim_info, stim_ids = comh.loadStimulusInfo(mat_dir)
        h5_file_list = comh.getFileListFromTrialIndices(h5_dir, stim_info[stim_info['stim_ids'] != 17].index.values, args.bin_width, args.window_size)
        unstimulated_log_likes, stimulated_log_likes = comh.getLikelihoodsForRegion(h5_file_list, args.region)
        comh.plotLogLikelihoodsHistograms(unstimulated_log_likes, args.region, image_dir, args.bin_width, title='unstimulated')
        comh.plotLogLikelihoodsHistograms(stimulated_log_likes, args.region, image_dir, args.bin_width,args.bin_width,  title='stimulated')
