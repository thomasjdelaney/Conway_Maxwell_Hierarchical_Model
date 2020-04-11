"""
For loading the data from a given neuropixels probe. Instructions taken from the script:
    http://data.cortexlab.net/dualPhase3/data/script_dualPhase3.m
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
"""
import os, argparse
import numpy as np
import pandas as pd
from scipy.io import loadmat
from collections import Counter
from scipy.sparse import csr_matrix

parser = argparse.ArgumentParser(description='Load the data from a given neuropixels probe.')
parser.add_argument('-p', '--probe_dir', help='Directory of the probe data.', default='posterior', choices=['posterior', 'frontal'])
parser.add_argument('-d', '--debug', help='Flag to enter debug mode.', action='store_true', default=False)
parser.add_argument('-s', '--save_cell_info', help='Flag to save the cell info to a csv.', action='store_true', default=False)
args = parser.parse_args()

proj_dir = os.path.join(os.environ['HOME'], 'Conway_Maxwell_Hierarchical_Model')
csv_dir = os.path.join(proj_dir, 'csv')
probe_dir = os.path.join(proj_dir, args.probe_dir)

def getProbeInfoDict():
    probe_info = loadmat(os.path.join(proj_dir, 'posterior', 'forPRBimecP3opt3.mat'))
    probe_info['connected'] = probe_info['connected'].flatten().astype(bool)
    probe_info['ycoords'] = probe_info['ycoords'].flatten()
    probe_info['xcoords'] = probe_info['xcoords'].flatten()
    probe_info['connected_ycoords'] = probe_info['ycoords'][probe_info['connected']]
    probe_info['connected_xcoords'] = probe_info['xcoords'][probe_info['connected']]
    return probe_info

def getClusterAveragePerSpike(spike_clusters, quantity):
    unique_clusters, spike_counts = np.unique(spike_clusters, return_counts=True)
    cluster_indices = np.array(list(map(lambda x:np.where(unique_clusters == x)[0][0], spike_clusters)))
    summation_over_clusters = csr_matrix((quantity, (cluster_indices, np.zeros(spike_clusters.size, dtype=int))), dtype=float).toarray().flatten()
    cluster_average = np.divide(summation_over_clusters, spike_counts)
    return cluster_average

def getTemplatePositionsAmplitudes(templates, whitening_matrix_inv, y_coords, spike_templates, template_scaling_amplitudes):
    num_templates, num_timepoints, num_channels = templates.shape
    unwhitened_template_waveforms = np.array(list(map(lambda x:np.matmul(x, whitening_matrix_inv), templates)))
    template_channel_amplitudes = np.max(unwhitened_template_waveforms, axis=1) - np.min(unwhitened_template_waveforms, axis=1)
    template_amplitudes_unscaled = np.max(template_channel_amplitudes, axis=1)
    threshold_values = template_amplitudes_unscaled*0.3
    for i in range(num_templates):
        template_channel_amplitudes[i][template_channel_amplitudes[i]<threshold_values[i]] = 0
    absolute_template_depths = np.array([template_channel_amplitudes[:,i]*probe_info['connected_ycoords'][i] for i in range(num_channels)]).sum(axis=0)
    template_depths = np.divide(absolute_template_depths, template_channel_amplitudes.sum(axis=1))
    spike_amplitudes = np.multiply(template_amplitudes_unscaled[spike_templates], template_scaling_amplitudes)
    average_template_amplitude_across_spikes = getClusterAveragePerSpike(spike_templates, spike_amplitudes)
    template_ids = np.sort(np.unique(spike_templates))
    template_amplitudes = np.zeros(np.max(template_ids+1), dtype=float)
    template_amplitudes[template_ids] = average_template_amplitude_across_spikes
    spike_depths = template_depths[spike_templates]
    waveforms = np.zeros(templates.shape[0:2], dtype=float)
    max_across_time_points = np.array(np.abs(templates)).max(axis=1)
    for i in range(num_templates):
        template_row = max_across_time_points[i,:]
        waveforms[i,:] = templates[i,:,template_row.argmax()]
    min_values, min_indices = [waveforms.min(axis=1), waveforms.argmin(axis=1)]
    waveform_troughs = np.unravel_index(min_indices, waveforms.shape)[1]
    template_duration = np.array([waveforms[i,waveform_troughs[i]:].argmax()for i in range(num_templates)])
    spike_amplitudes = spike_amplitudes*0.6/512/500*1e6
    return spike_amplitudes, spike_depths, template_depths, template_amplitudes, unwhitened_template_waveforms, template_duration, waveforms

def makeCellInfoTable(cluster_groups, cluster_depths, cluster_amplitudes, probe, id_adjustor=0):
    thalamus_threshold = 1634.0
    hippocampus_threshold = 2797.0
    v1_threshold = 3840.0
    motor_cortex_threshold = 3840.0
    striatum_threshold = 1550.0
    if probe == 'posterior':
        regions = np.repeat('thalamus', cluster_groups.size).astype(object)
        regions[cluster_depths > thalamus_threshold] = 'hippocampus'
        regions[cluster_depths > hippocampus_threshold] = 'v1'
    if probe == 'frontal':
        regions = np.repeat('striatum', cluster_groups.size).astype(object)
        regions[cluster_depths > striatum_threshold] = 'motor_cortex'
    cell_info = pd.DataFrame({'group':cluster_groups, 'depth':cluster_depths, 'amplitude':cluster_amplitudes, 'region':regions, 'probe':probe}, index=cluster_groups.index)
    cell_info.index = cell_info.index + id_adjustor
    cell_info_file = os.path.join(proj_dir, 'csv', 'cell_info.csv')
    if not(os.path.isfile(cell_info_file)):
        cell_info.to_csv(cell_info_file, index_label='cluster_id')
    else:
        cell_info.to_csv(cell_info_file, index_label='cluster_id', header=False, mode='a')
    return pd.read_csv(cell_info_file, index_col='cluster_id')

probe_info = getProbeInfoDict()
cluster_groups = pd.read_csv(os.path.join(probe_dir, 'cluster_groups.csv'), sep='\t', index_col='cluster_id')['group']
cluster_ids = np.array(cluster_groups.index)
noise_clusters = cluster_ids[cluster_groups == 'noise']
spike_clusters = np.load(os.path.join(probe_dir, 'spike_clusters.npy'))
not_noise_indices = ~np.in1d(spike_clusters, noise_clusters)
spike_times = np.load(os.path.join(probe_dir, 'spike_times.npy')).flatten()[not_noise_indices]
frames_per_second = 30000.0
spike_seconds = spike_times/frames_per_second
spike_templates = np.load(os.path.join(probe_dir, 'spike_templates.npy')).flatten()[not_noise_indices]
template_scaling_amplitudes = np.load(os.path.join(probe_dir, 'amplitudes.npy')).flatten()[not_noise_indices]
spike_clusters = spike_clusters[not_noise_indices]
cluster_ids = np.setdiff1d(cluster_ids, noise_clusters)
cluster_groups = cluster_groups[cluster_ids]
templates = np.load(os.path.join(probe_dir, 'templates.npy'))
whitening_matrix_inv = np.load(os.path.join(probe_dir, 'whitening_mat_inv.npy'))
spike_amplitudes, spike_depths, template_depths, template_amplitudes, unwhitened_template_waveforms, template_duration, waveforms = getTemplatePositionsAmplitudes(templates, whitening_matrix_inv, probe_info['connected_ycoords'], spike_templates, template_scaling_amplitudes)
cluster_depths = getClusterAveragePerSpike(spike_clusters, spike_depths)
cluster_amplitudes = getClusterAveragePerSpike(spike_clusters, spike_amplitudes)
if args.probe_dir == 'frontal':
    time_correction = np.load(os.path.join(probe_dir, 'time_correction.npy'))
    spike_times = spike_times - time_correction[1]
    cell_info_file = os.path.join(csv_dir, 'cell_info.csv')
    if os.path.exists(cell_info_file):
        posterior_cell_info = pd.read_csv(os.path.join)
        id_adjustor = posterior_cell_info['cluster_id'].max() + 1
    else:
        print("WARNING: cell_info.csv doesn't yet exist.")
if args.save_cell_info:
    cell_info = makeCellInfoTable(cluster_groups, cluster_depths, cluster_amplitudes, args.probe_dir)
else:
    cell_info = pd.read_csv(os.path.join(proj_dir, 'csv', 'cell_info.csv'), index_col='cluster_id')
