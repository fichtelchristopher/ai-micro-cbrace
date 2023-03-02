''' 
Master thesis project of Christopher Fichtel (christopher.fichtel@ottobock.com) started in Oct 2022
AI in operating ortheses

File created 07 Oct 2022

Preprocessing of input data. 
'''
import numpy as np
import datetime
from datetime import datetime, timezone
from Configs.constants import *
import matplotlib.pyplot as plt
from Configs.namespaces import *
from Misc.math import * 
from collections import Counter
from Misc.utils import *

datetime_format = '%H:%M:%S:%f'

def preprocess_df(df, knee_lever = None, aos_data = True):
    '''
    Perform preprocessing steps. Calling respective functions.

    :param knee_lever: optional, loaded from Kinematics.mat file   
                        if provided, the knee moment will be calculated and saved as a column "KNEE_MOMENT"
    '''
    df = cut_off_starting_time(df)

    if not aos_data:
        add_cycle_time_col(df) 

    df = add_sampling_freq_col(df)
    df = add_time_in_s(df) 
    # df = add_measurement_idx(df)

    if knee_lever is not None:
        df = add_knee_moment(df, knee_lever)
    df = preprocess_column_datatypes(df)
    df = add_total_acc(df)
    df = add_activity_idx(df)

    df = add_index(df)
    
    return df

def add_cycle_time_col(df, fsampling_hz = 100):
    '''
    Add cycle time column, indicates sampling step in ms.
    :param fsampling: the sampling frequency in Hz
    '''
    step_size_ms = 1000 / fsampling_hz

    df[CYCLETIME_COL] = step_size_ms

    return df


def add_activity_idx(df):
    '''
    Add an activity idx column with zeros (for the beginning)
    '''
    df[ACTIVITY_IDX_COL] = np.zeros(len(df))

    return df

def add_index(df):
    '''
    Add the INDEX_COL column.
    '''
    df[INDEX_COL] = df.index
    
    return df

def cut_off_starting_time(df):
    '''
    Each file has a few data samples (~10) at the beginning where recording did not start yet. Cut these samples of.

    :return: df.iloc[starting_point:]
    '''
    if "BATTERY_CHARGE" not in df.columns.values: 
        return df

    for index, val in enumerate(df["BATTERY_CHARGE"]): 
        if val > 0:
            return df.iloc[index:]

        if index > 100:
            break

    return df

# def add_measurement_idx(df):
#     '''
#     One file consists of multiple measurements. Add a measurement index file based on temporal distances following the real time clock.
#     Caution: the format of real time clock varies from file to file (mat - csv file different) 
#     '''
#     assert(CYCLETIME_COL in df.columns.values)
    
#     time = df["RTC_HHMMSSZS"].values

#     time_dif = time[1:] - time[:-1]

#     c = Counter(time_dif) 
#     c = c.most_common(2)
#     threshold = (c[0][0] + c[1][0]) / 2 * 10 # treshold in milliseconds considered a change in the measurement --> should correspond to around 0.5
#     counter_jump_indices = np.where(time_dif > threshold)[0] + 1

#     measurement_idx = np.zeros(len(time), dtype = int)

#     for counter, i in enumerate(counter_jump_indices):
#         measurement_idx[i:] = counter + 1

#     df["MEASUREMENT_IDX"] = measurement_idx

#     return df


def add_sampling_freq_col(df):
    '''
    :param df: dataframe that contains the column CYCLETIME_COL
    :return: dataframe containing an additional SAMPLING_FREQ_COL column -->  SAMPLING_FREQ=10    for CYCLETIME=100 and
                                                                            SAMPLING_FREQ=100   for CYCLETIME=10 
    '''
    df[SAMPLING_FREQ_COL] = 1000 / df[CYCLETIME_COL] # CYCLETIME indicates the step in ms to the next sample
    return df


def add_time_in_s(df):
    '''
    :param df:  dataframe that contains the sampling frequency in column SAMPLING_FREQ_COL
    :return:    dataframe containing an additional TIME_SEC_COL column with time indiciated in seconds as float
    '''
    df[TIME_SEC_COL] = df[CYCLETIME_COL].cumsum() / 1000

    return df


    # # FOR .MAT AND .CSV FILES
    # if "RTC_HHMMSSZS" in df.columns.values:
    #     time = df["RTC_HHMMSSZS"].values

    #     if time.dtype != float:\questions
    #         # time = [ t.split(":") for t in time]
    #         time = [datetime.strptime(t, datetime_format) for t in time]
    #         time = np.array([t.replace(tzinfo=timezone.utc).timestamp() for t in time])

    #     time = time - np.min(time)

    #     df[TIME_SEC_COL] = time

    # # FOR .TXT FILES
    # elif "Time\n" in  df.columns.values:

    #     time = df["Time\n"].values
    #     time = [int(t, 16) for t in time]

    #     # time is in samples assuming the sampling rate is 100 Hz
    #     sampling_rate = 100
    #     time = [t / sampling_rate for t in time]
        
    # else:
    #     print("Could not add time in s to dataframe.")
    #     return df
    # plt.plot(df["CYCLECOUNTER"])
    # cycle_counter_dif = df["CYCLECOUNTER"][1:].values - df["CYCLECOUNTER"].values[:-1]

    # plt.plot(time)
    # # plt.plot(time_dif, color = "blue")
    # plt.show()
    # plt.vlines(counter_jump_indices, ymin = 0, ymax = 10000, color = "red")

    return df

def add_knee_moment(df, knee_lever):
    '''
    Knee moment added and used for activity statistics
    :param df:  dataframe loaded via the load_file()-method
    :return:    dataframe containing an additional "KNEE_MOMENT" column
    '''
    # TODO revise computation

    # original matlab implementation (from AOS_Batch_Analysis-script):
    # M_K=AOSdata(:,col.FH)*cal_FH.*knee_lever(int16(AOSdata(:,col.K)*10)+96,2); 
    # python implementation
    load_factor = cal_FH * df["JOINT_LOAD"].values
    df["JOINT_ANGLE"] = df["JOINT_ANGLE"].replace(",", ".", regex = True).astype("float64")
    knee_lever_indices = np.array(df["JOINT_ANGLE"].values * 10, dtype = int) + 96
    angle_factor = knee_lever[:, 1][knee_lever_indices]
    knee_moment = np.multiply(load_factor, angle_factor)    # element wise multiplication

    df = df.assign(KNEE_MOMENT = knee_moment)

    return df

def add_total_acc(df):
    '''
    Total acceleration computation based on acceleration in x, y and z direction

    :param df:  dataframe that contains the ACC_COLS (see namespaces)
    :return:    dataframe containing an TOTAL_ACC_COL (see namespaces) 
    '''
    total_acc = get_total_acceleration(df)

    df = df.assign(DDD_ACC_TOTAL = total_acc)
 
    return df

def preprocess_column_datatypes(df):
    '''
    Change dtype of rotation, acceleration and velocity columns into float
    The 9 respective columns start with "DDD" e.g. "DDD_VEL_Z" or "DDD_YAW"
    :param df: input df 
    :return: dataframe with adjusted datatypes
    '''
    float_cols = [c for c in df.columns.values if "DDD" == c[:3]]
    int_cols = [c for c in df.columns.values if "RI_" == c[:3]]

    df[float_cols] = df[float_cols].replace(",", ".", regex = True)
    df[float_cols] = df[float_cols].astype("float64")

    df[int_cols] = df[int_cols].astype("float64")

    return df