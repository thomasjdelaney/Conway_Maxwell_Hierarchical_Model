import os
import pandas as pd
import numpy as np
from scipy.io import loadmat

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

