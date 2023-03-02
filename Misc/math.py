''' 
Master thesis project of Christopher Fichtel started in Oct 2022
AI in operating ortheses

File created 06 Oct 2022

Code Base for mathematical operations
'''
import numpy as np
from Configs.namespaces import *
import matplotlib.pyplot as plt
import scipy
import scipy.interpolate


def rsm(arr):
    '''
    Determine the root sum quare, e.g. for calculating total acceleration based on the 3 axis
    
    :param arr: shape of [sequence_length, dimensions] e.g. 1000 x 3 (3 for the dimensions x,y,t) 
    :return: rsm value per "point in time" e.g. 1000 x 1
    '''

    rsm = np.square(arr)

    rsm = np.sum(rsm, axis = 1)
    rsm = np.sqrt(rsm)

    return rsm

def rsm_df(df):
    '''
    Determine the root sum quare, e.g. for calculating total acceleration based on the 3 axis
    
    :param df: dataframe containing only the columns for rms computation   
    :return: array, 1 rsm value per "point in time", resulting array has same length as the original 
    '''
    arr = df.to_numpy()
    rms_arr = rsm(arr)

    return rms_arr

def get_total_acceleration(df):
    '''
    :param df: pandas dataframe containing the acceleration columns (DDD_ACC_X, DDD_ACC_Y, DDD_ACC_Z)
    :return: df with an additional 
    '''

    assert(set(ACC_COLS).issubset(set(list(df.columns.values))))

    total_acc = rsm_df(df[ACC_COLS])

    return total_acc

def get_batchwise_var(df, batch_size):
    '''
    Compute the batch wise variance.

    :param df:      filtered df that only contains columns used for variance
    :batch_size:    size of chunks for which variance is computed

    '''
    batch_var = []

    for start_idx in range(0,len(df), batch_size):
        end_idx = np.amin((start_idx + batch_size , len(df)))
        df_acc_batch = df[start_idx:end_idx]
        df_acc_batch_var = get_batch_var(df_acc_batch)
        batch_var.extend([df_acc_batch_var] * (end_idx-start_idx))

    return batch_var

def get_batch_var(df_batch):
    '''
    Computation of a variance value for a batch of values.
    1. Absolute value generation
    2. Computation of standard deviation for every column
    3. Summing up individual standard deviations

    :param df_batch: the batch dataframe (or a "normal" dataframe) for which the variance should be computed
    :return: variance value
    '''
    df_batch = np.abs(df_batch)
    df_batch_var = np.std(df_batch, axis = 0)
    df_batch_var = np.sum(df_batch_var)
    return df_batch_var


def interpolate(input_array, output_len = 101): 
    ''' 
    Interpolate the given array to a certain output_len

    :param input_array: array to be "scaled"
    :param output_len: 101 standard to scale input array to percentage of gait cycle

    :return: interpolated array of len 101
    '''

    x_new = np.arange(0, output_len, 1) # one value per percent of the gait cycle from 0 to 100

    linspace = np.linspace(0, output_len - 1, len(input_array))
    f = scipy.interpolate.interp1d(linspace, input_array)

    output_array = f(x_new)

    return output_array