''' 
Master thesis project of Christopher Fichtel started in Oct 2022
AI for microprocessor-controlled knee joints

File created 09 Nov 2022

Code Base for tensorflow dataset utils
'''
import tensorflow as tf
import numpy as np
from Configs.namespaces import * 
from Configs.pipeline_config import *
from Configs.nn_pipeline_config import *
from Configs.shallow_pipeline_config import *
from Visualisation.visualisation import *
from Misc.utils import *
import random
from scipy.stats import norm
from Gait.steps_utils import get_subject_step_transitions_from_activity_progression

from sklearn.preprocessing import MinMaxScaler

from scipy.interpolate import interp1d

from pathlib import Path

from Processing.loading_saving import load_from_pickle, save_as_pickle

from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import sys

from Analysis.activity_analysis import get_label_distr_dict, get_label_distr_dict_from_activity_list


def get_labels_from_dataset(dataset,  activity_code_type_dict = ACTIVITY_CODE_TYPE_DICT_CHRIS_CONV):
    '''
    :param dataset: a tf dataset generator object.
    '''
    one_hot_encoder = get_one_hot_encoder(activity_dict=activity_code_type_dict)
    labels_one_hot = [x[1] for x in dataset.unbatch().as_numpy_iterator()]
    labels = one_hot_encoder.inverse_transform(labels_one_hot)[:, 0]

    return labels

def get_label_distribution_from_dataset(dataset, activity_code_type_dict = ACTIVITY_CODE_TYPE_DICT_CHRIS_CONV):
    '''
    :param dataset: a tf dataset generator object.
    '''
    labels = get_labels_from_dataset(dataset, activity_code_type_dict)

    label_distr = get_label_distr_dict(activity_code_type_dict, labels)

    return label_distr

def apply_mapping(data_df, mapping):
    '''
    '''
    data_df =  data_df.replace({LABEL_COL: mapping})

    return data_df

def create_reduced_df(data_df,
                    activity_code_type_dict, # the mapping from integer to string
                    max_activity_duration_sec = MAX_ACTIVITY_DURATION_SEC,
                    activity_transition_time_sec = ACTIVITY_TRANSITION_TIME_SEC,
                    visualize_cutting_area = VISUALIZE_CUTTING_AREA,
                    mapping = None):
    '''
    Reduce data: "throw away" within a time frame for activities that don't change over a long period of time (e.g. 3 minutes)
        --> we are especially interested in the transitions and not activities that dont change (e.g. sitting for several minutes)  
    
    :param data_df: must contain ACTIVITY_COL and SAMPLING_FREQ_COL-columns
    :param max_activity_duration_sec X: activities longer X SECONDS will be cropped
    :param activity_transition_time_sec Y: 
        in seconds: if an activity is found that doesn't change over max_activity_duration SECONDS, "cut" activity 
        --> throw away data in the range [activitiy start + Y seconds, activity end - Y seconds]
    
    :param output_activity_code_type_dict: the code-type dictionary used in the reduced dataframe
        --> integer code as key and string class label as item 

    :param mapping: the transofrmation for the input labels in data_df e.g. the entry  100: -100 means that the label 100 in the 
                    datat_df-label column will be transformed to -100
                    see the calling function for more information (reduce_data)
                

    :return: dataframe where the "uninteresting" data is thrown away and additional column "ACTIVITY_IDX"
    '''
    initial_length_samples = len(data_df)

    assert(TIME_SEC_COL in data_df.columns.values)

    if type(mapping) is not type(None):
        data_df = data_df.replace({LABEL_COL: mapping})


    # Optional visualisation of the input ground truth file
    # plot_default_gt(data_df)
    # plt.grid(visible = True, axis = "y")
    # plt.legend()
    # plt.title(os.path.basename("test"))
    # plt.show()
    # plt.clf()


    activity_progression = data_df[LABEL_COL].iloc[1:].values - data_df[LABEL_COL].iloc[:-1].values

    indices = np.where(activity_progression != 0)[0]

    indices = np.insert(indices, 0, -1) 
    indices = np.append(indices, -1)
    
    cut_activity_ranges = []

    counter = 0

    for activity_start_idx, activity_end_idx in zip(indices[:-1], indices[1:]):
        activity_start_idx += 1

        if activity_end_idx == -1:
            activity_end_idx = len(data_df)
        
        # Throw away activities of discarded class
        class_code = data_df[LABEL_COL].iloc[activity_start_idx]
        if class_code == DISCARD_CLASS_CODE:
            cut_activity_ranges.append((activity_start_idx, activity_end_idx+1))
            counter += 1
            continue

        fsampling = data_df[SAMPLING_FREQ_COL].iloc[activity_start_idx]

        if activity_end_idx == len(data_df):
            duration_sec = data_df[TIME_SEC_COL].iloc[activity_end_idx-1] - data_df[TIME_SEC_COL].iloc[activity_start_idx]
        else:
            duration_sec = data_df[TIME_SEC_COL].iloc[activity_end_idx] - data_df[TIME_SEC_COL].iloc[activity_start_idx]

        if duration_sec > max_activity_duration_sec:

            activity_transition_time_samples = fsampling * activity_transition_time_sec

            if activity_start_idx + activity_transition_time_samples < (activity_end_idx - activity_transition_time_samples):
                counter += 1
                cut_activity_ranges.append((int(activity_start_idx + activity_transition_time_samples),
                                        int(activity_end_idx - activity_transition_time_samples)))


    # Visualize cutting areas
    # visualize_cutting_area = False
    # if visualize_cutting_area:
    #     data_cols = [ACTIVITY_COL, "JOINT_ANGLE", "DDD_ACC_TOTAL"]
    #     plot_cutting_areas(data_df, data_cols, cut_activity_ranges, data_scaling_factors = [1/10, 1, 1])
    #     plt.show()


    activity_idx = np.zeros(len(data_df))

    idx = 1
    for start_idx, end_idx in cut_activity_ranges:
        activity_idx[end_idx:] = idx
        idx += 1

    data_df[ACTIVITY_IDX_COL] = activity_idx

    visualize_cutting_area = False
    if visualize_cutting_area:
        matplotlib.use("TkAgg")
        data_cols = [LABEL_COL, ACTIVITY_IDX_COL, "JOINT_ANGLE", "KNEE_MOMENT"]
        plot_cutting_areas(data_df, data_cols, cut_activity_ranges, data_scaling_factors = [1/10, 10, 1, 1])
        plt.legend()
        plt.show()
        matplotlib.use("Agg")

    if len(cut_activity_ranges) > 0:
        data_df_out = data_df.iloc[:cut_activity_ranges[0][0]]
        
        end_idx_post = -1

        for (start_idx_prev, end_idx_prev), (start_idx_post, end_idx_post) in zip(cut_activity_ranges[:-1], cut_activity_ranges[1:]):
            data_df_out = pd.concat([data_df_out, data_df.iloc[end_idx_prev:start_idx_post]])

        if end_idx_post > 0:
            data_df_out = pd.concat([data_df_out, data_df.iloc[end_idx_post:]])
    else: 
        data_df_out = data_df

    # plot_signals_from_df(data_df, data_cols = [ACTIVITY_COL, "ACTIVITY_IDX"], data_scaling_factors = [1, 100])
    # plt.show()

    out_length_samples = len(data_df_out)

    visualize_cutting_area = False
    if visualize_cutting_area:
        for activity_idx in list(set(data_df_out[ACTIVITY_IDX_COL].values)):
            data_df_indexed = data_df_out[data_df_out[ACTIVITY_IDX_COL] == activity_idx]
            plt.plot(data_df_indexed["DDD_ACC_TOTAL"]) # plot some signal
            plt.plot(data_df_indexed["ACTIVITY_IDX"])
        plt.show()   

    return data_df_out

def get_one_hot_encoder(activity_dict = ACTIVITY_CODE_TYPE_DICT_CHRIS):
    '''
    The activity dict keys hold the class labels 
    :return: a one hot encoder, apply to class labels via one_hot_encoder.transform( ... )
    '''
    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoder.fit(np.array(list(activity_dict.keys())).reshape(-1, 1))
    return one_hot_encoder


def get_one_hot_encoder_from_labels(label_list):
    '''
    :param label_list: all possible labels, sorted 
                    i.e. label at idx 0 will be in column 0 and so on
    '''
    label_list = list(label_list)
    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoder.fit(np.array(label_list).reshape(-1, 1))
    return one_hot_encoder


def get_windowed_tf_dataset_from_indices(data_df, signal_cols, label_col, window_size_samples, batch_size, end_indices, activity_code_type_dict = ACTIVITY_CODE_TYPE_DICT_CHRIS):
    '''
    We create time windows to create X and y features.
    For example, if we choose a window of 30, we will create a dataset formed by 30 points as X

    :param data_df: from this dataframe extract the signals and label 
                        label will be converted to a one hot encoded dataframe
    :param signal_cols: list of of signal cols to be used 
    :parm label_col: the label column
    :param window_size_samples: window size in samples. Sampling freq is 100 Hz --> 100 Samples correspond to 1 seconds
    :param window_step_size: shift of the window in samples
    :param batch_size: -
    :param shuffle_buffer: - 
    :param activity_dict: the dictionary containing the class labels as keys --> used for one hot encoding 
    :param shuffle: whether to shuffle the instances --> should be false for validation set
    :param indices: the "end" indices of each window --> these are NOT the indices from data_df.index but those you can access via iloc[]
    '''
    # end_indices = end_indices #- np.min(end_indices) + window_size_samples
    start_indices = np.array(end_indices) - window_size_samples

    df_X = data_df[signal_cols] # data dataframe
    df_Y = data_df[label_col]   # label dataframe

    # One Hot encode the label dataframe
    one_hot_encoder = get_one_hot_encoder(activity_dict=activity_code_type_dict)
    df_Y = pd.DataFrame(data=one_hot_encoder.transform(np.array(df_Y).reshape(-1 , 1)), columns=list(one_hot_encoder.categories_[0])).set_index(df_Y.index)
    n_classes = len(df_Y.columns.values)

    # Concat data and label and make sure one hot encoded is on the "right side" of the dataframe
    data = np.empty(shape = (len(end_indices), window_size_samples, len(df_X.columns.values)))
    labels = np.empty(shape = (len(end_indices), n_classes))

    df_X = np.array(df_X)
    df_Y = np.array(df_Y)

    # for start, end in zip(start_indices, end_indices):
    #     if end-start-window_size_samples != 0:
    #         print("\n\n\n\n")
    #         print(start)
    #         print(end)
    #         print("\n\n\n\n")
    # print("\n\n\n\n\n")
    # print(batch_size)
    # print("\n\n\n\n\n")
    data = np.array([df_X[start_idx:end_idx, :] for start_idx, end_idx in zip(start_indices, end_indices)])
    labels = np.array([df_Y[end_idx] for end_idx in end_indices])

    data_ds = tf.data.Dataset.from_tensor_slices(data)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)

    dataset = tf.data.Dataset.zip((data_ds, labels_ds))

    # if shuffle:
    #     dataset = dataset.shuffle(1000)

    # Apply the mapping function
    dataset_out = dataset.batch(batch_size).prefetch(1)

    # for c, (x, y) in enumerate(dataset_out):
    #     if c > 1:
    #         break
        # print("Output data shape")
        # print(x.numpy().shape)
        # print("Output label shape")
        # print(y.numpy().shape)

    return dataset_out

def train_val_split(Y, data_df, split_idx):
	'''
    Divide the time series into training and validation set
    :param Y: labels
    :param data_df: dataframe with the data
    :split_idx: integer value indicating where to split the data

    :return: 
    '''
	Y_train = Y[:split_idx]
	data_train = data_df[:split_idx]
	Y_val = Y[split_idx:]
	data_val = data_df[split_idx:]

	return Y_train, data_train, Y_val, data_val

def encode_one_hot(labels_arr, labels):
    '''
    Return dataframe with one hot label columns
    '''
    labels_df = pd.DataFrame({"label": labels_arr})

    labels_df = pd.get_dummies(labels_df["label"])

    for label in labels:
        if label not in labels_df.columns.values:
            labels_df[label] = np.zeros(len(labels_df))

    return labels_df[labels]

def scale_min_max(train_df, feature_cols,  val_df = None, min_val = 0, max_val = 1):
    '''
    Scale the features to be in the range of (min_val, max_val)
    Split, scale your training data, then use the scaling from your training data on the testing data.

    if no val_df param is submitted only return the scaled train df. If it is submitted, apply the same
    transform on the val_df as on the train_df
    '''
    scaler = MinMaxScaler(feature_range=(min_val, max_val))

    scaler = scaler.fit(train_df[feature_cols])

    train_df[feature_cols] = scaler.transform(train_df[feature_cols])

    if type(val_df) == type(None):
        return train_df
    else:   
        val_df[feature_cols] = scaler.transform(val_df[feature_cols])
        return train_df, val_df

def scale_standard(train_df, feature_cols,  val_df = None):
    '''
    Remove mean and standard deviation to 1
    '''
    scaler = StandardScaler()

    scaler = scaler.fit(train_df[feature_cols])

    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    if type(val_df) == type(None):
        return train_df
    else:  
        val_df[feature_cols] = scaler.transform(val_df[feature_cols])
        return train_df, val_df

def balance_dataset(X_data, Y_labels, mode = "standard"):
    '''
    This method balances the dataset in that sense that it reduces the occurences of the most prominent class (only!). It
    does not change the occurences of the 2nd, 3rd and so on most prominent classes. See the 2 modes below for further
    explanation.

    :param mode:
        "standard" -->  This mode is especially suited for cases with one highly overrepresented class.
                        if the overrepresented class' ratio is above 0.5 --> set the sampling so that the overrepresented class roughly has 0.5 ratio 
                        if overrepresented ratio < 0.5 --> same as "approach_second" mode

        "approach_second": set sampling so that overrepresented class roughly has the ratio has the 2nd most present class

        ""
    '''
    value_counts = Y_labels.value_counts()
    overrepresented_class = value_counts.argmax()
    max_represented_classes = value_counts.index.tolist() # sorted from top --> down

    overrepresented_ratio = value_counts[overrepresented_class] / len(Y_labels) 
    if (overrepresented_ratio < 0.5) or (mode == "approach_second"):
        # Approach to next biggest class
        overrepresented_ratio = 1 - (value_counts[max_represented_classes[1]] / len(Y_labels))
            
    underrepresented_ratio = 1 - overrepresented_ratio

    num_jump_samples = int(overrepresented_ratio / underrepresented_ratio)

    overrepresented_indices = np.where(np.array(Y_labels) == overrepresented_class)[0]
    underrepresented_indices = np.where(np.array(Y_labels) != overrepresented_class)[0]

    overrepresented_indices = overrepresented_indices[::num_jump_samples]

    keep_indices = list(underrepresented_indices) + list(overrepresented_indices)
    
    if type(X_data) == type(pd.DataFrame):
        X_data_balanced = X_data.iloc[keep_indices]
    else:
        X_data_balanced = X_data[keep_indices]
    Y_labels_balanced = np.array(Y_labels)[np.array(keep_indices, dtype = int)]

    return X_data_balanced, Y_labels_balanced

def get_importance_df(distribution_dict, ouptut_feature_range = (0, 1)):
    '''
    # TODO Apply some transformations to get an importance dictionary. Key is the label name and 
    Importance values scaled to lie in [0.0, 1.0]
    

    :param type:    "activity" analyses the activty values only and not the transitions, returns a dictionary with the importances between (0, 1)
                    "transition" analyses the frequency of transitions in which a certain activity takes part 

    '''
    activity_importance_df = pd.DataFrame()
    for activity, num in distribution_dict.items():
        activity_importance_df = activity_importance_df.append({"activity": activity, "num": num}, ignore_index = True)

    num_samples_total = sum(distribution_dict.values())

    activity_importance_df["importance"] = activity_importance_df["num"] / num_samples_total
    activity_importance_df["importance"] = 1 / activity_importance_df["importance"]
    activity_importance_df["importance"] = np.log(activity_importance_df["importance"])
    activity_importance_df["importance"] = np.exp(activity_importance_df["importance"])
    # activity_importance_df["importance"] *= activity_importance_df["importance"]
    # visualise_activity_importance(activity_importance_df)

    scaler = MinMaxScaler(feature_range=ouptut_feature_range)
    importance_arr = np.array(activity_importance_df["importance"].values).reshape(-1, 1)
    scaler.fit(importance_arr)
    activity_importance_df["importance"] = scaler.transform(importance_arr)[:, 0]    

    return activity_importance_df


def get_activity_sampling_probability(activity_importance_df, sampling_prob_min_transition, sampling_prob_max_transition):
    '''
    Use the activity_importance_df returned from get_importance_dict
    '''
    # set the minimum transition probability
    scaler = MinMaxScaler(feature_range=(sampling_prob_min_transition, sampling_prob_max_transition))
    importance_arr = np.array(activity_importance_df["importance"].values).reshape(-1, 1)
    scaler.fit(importance_arr)
    activity_importance_df["activity_sampling_probability"] = scaler.transform(importance_arr)

    sampling_prob_dict = df_to_dict(activity_importance_df, "activity", "activity_sampling_probability")

    return sampling_prob_dict

def add_sampling_prob_col(data_df, default_step_size, window_size, label_col = LABEL_COL, feature_file = False, activity_distribution_dict = None, transition_distribution_dict = None):
    '''
    :param label_distribution_dict: use submitted dict if submitted, otherwise label distribution dict will be calculated on the labels in this dataframe data_df
    :param transition_distr_dict: use submitted dict if submitted, otherwise transition distribution dict will be calculated on the labels in this dataframe data_df
    #TODO currently transition df not used
    '''
    # activity_distribution_dict = None
    # transition_distr_dict = None

    f_sampling_default = 1 / default_step_size
    sampling_prob_min_transition = 4*f_sampling_default
    sampling_prob_max_transition = 1.0

    sampling_prob_output = np.full(shape=(len(data_df)), fill_value = f_sampling_default)
    if default_step_size == 1: 
        if not feature_file:
            sampling_prob_output[:window_size] = -1
        data_df["sampling_prob"] = sampling_prob_output
        return data_df

    assert(sampling_prob_min_transition < sampling_prob_max_transition)
    
    # compute the distribution dict on the current dataframe
    activity_progression = data_df[label_col].values

    # Per activity, get the ratio of transition "participations"
    if type(transition_distribution_dict) == type(None):
        # Get also a transition importance dictionary 
        transitions, transition_df = get_subject_step_transitions_from_activity_progression(activity_progression, discard_swext=False)
        transition_distribution_dict = {}
        for activity in list(set(list(transition_df["from"].values))):
            transition_distribution_dict[activity] = sum(transition_df.loc[transition_df["from"] == activity]["num"])

    if type(activity_distribution_dict) == type(None):
        activity_distribution_dict = get_label_distr_dict_from_activity_list(activity_progression) 

    # Importances scaled to a value of 0 and 1
    transition_importance_df = get_importance_df(transition_distribution_dict, ouptut_feature_range=(sampling_prob_min_transition, sampling_prob_max_transition))
    activity_importance_df = get_importance_df(activity_distribution_dict)

    transition_importance_dict  =   df_to_dict(transition_importance_df, "activity", "importance")
    activity_importance_dict    =   df_to_dict(activity_importance_df, "activity", "importance")

    activity_sampling_prob_dict = scale_dictionary(activity_importance_dict, sampling_prob_min_transition, sampling_prob_max_transition)
    
    gaus_fit_size_def = 256
    std_denominator = 16 # a lower value will increase the dataset 
    # # transition_gaus_fit_dict = scale_dictionary(transition_importance_dict, gaus_fit_size_def-gaus_fit_size_def*0.2, gaus_fit_size_def+gaus_fit_size_def)
    # transition_std_den_dict = scale_dictionary(transition_importance_dict, -16, -12)
    # for k, v in transition_std_den_dict.items():
    #     transition_std_den_dict[k] = v * (-1)

    activity_change_indices = np.where(activity_progression[1:]-activity_progression[:-1] != 0)[0]
    for activity_change_prev, activity_change_idx, activity_change_post in zip(activity_change_indices[:-2], activity_change_indices[1:-1], activity_change_indices[2:]):
        act_prev = activity_progression[activity_change_idx-1]
        act_post = activity_progression[activity_change_idx+1]

        trans_prob_prev = activity_sampling_prob_dict[act_prev]
        trans_prob_post = activity_sampling_prob_dict[act_post]
        
        gaus_fit_size_prev = gaus_fit_size_def #transition_gaus_fit_dict[act_prev]
        gaus_fit_size_post = gaus_fit_size_def #transition_gaus_fit_dict[act_post]

        start_idx   = int(np.max([activity_change_idx-gaus_fit_size_prev/2, activity_change_prev]))
        stop_idx    = int(np.min([activity_change_idx+gaus_fit_size_post/2, activity_change_post]))

        start_offset = activity_change_idx-start_idx
        stop_offset  = stop_idx - activity_change_idx

        # Gaus before transition
        std = gaus_fit_size_prev / std_denominator
        x_prev = np.arange(-start_offset, 0) 
        gaus_prev = norm.pdf(x_prev, 0, std)
        gaus_prev = (gaus_prev / np.max(gaus_prev)) * trans_prob_prev

        # Gaus after transition
        std = gaus_fit_size_post / std_denominator
        x_post = np.arange(0, stop_offset) 
        gaus_post = norm.pdf(x_post, 0, std)
        gaus_post = (gaus_post / np.max(gaus_post)) * trans_prob_post

        sampling_prob_idx = sampling_prob_output[start_idx:stop_idx]
        m = np.maximum(sampling_prob_idx, np.concatenate((gaus_prev, gaus_post)))

        sampling_prob_output[start_idx:stop_idx] = m

    if not feature_file:
        sampling_prob_output[:window_size] = -1
    data_df["sampling_prob"] = sampling_prob_output

    print("Calculating sampling probability done.")
    # For debugging
    # plot_default_gt(data_df, _plot_indices = False, data_cols = ["JOINT_ANGLE", LABEL_COL], data_labels=["ANGLE", LABEL_COL], data_scaling_factors=[1, 1/10])
    # plt.plot(data_df.index, sampling_prob_output * 100, label = "prob [%]")
    # plt.legend()
    # plt.show()

    return data_df 

def get_adaptive_sampling_indices(data_df, window_size, max_step_size, min_step_size = 1, f_sampling_min_factor = 5, gaus_fit_size = 512, gaus_std_fac = 8, label_col = LABEL_COL, feature_file = False):
    '''
    :param max_step_size: in regions with no activity changes, use this step size
                        --> defines the default sampling probability by 1/max_step_sizes
    :param f_sampling_min_factor: 
                        --> the most frequent transition (i.e. the "least" important) will still have hat least the factor 
                            higher sampling probabiltiy than the default one given by the max step size

    :param gaus_std_fac: a higher factor will make the gaussian more flat 
                        will be used to compute the std like: std = gaus_fit_size / gaus_std_fac
    '''
    random.seed(23)
    if not feature_file:
        data_df = data_df.iloc[window_size:]
    indices = [idx for idx, prob in zip(data_df.index, data_df["sampling_prob"]) if random.random() < prob]

    return indices

def get_data_labels_from_files(files, selected_feature_cols, is_training, step_size = 1, transformer = None, loc_indices_list = []):
    '''
    
    '''
    if len(loc_indices_list) != len(files):
        loc_indices_list = [None] * len(files)

    for c, (train_file, loc_indices) in enumerate(zip(files, loc_indices_list)):
        X_train_f, Y_train_f = get_data_labels_from_file(train_file, selected_feature_cols, is_training, step_size=step_size, data_transformer = transformer, loc_indices=loc_indices)
        if c == 0:
            X_train, Y_train = X_train_f, Y_train_f
        else:
            X_train, Y_train = np.concatenate((X_train, X_train_f), axis = 0), np.concatenate((Y_train, Y_train_f), axis = 0)

    return X_train, Y_train

def get_data_labels_from_file(filename, selected_feature_cols, is_training, label_col = LABEL_COL, _balance_dataset = True, step_size = 1, data_transformer = None, loc_indices = None):
    '''
    Based on the csv in the filename create X (data) and Y (labels) that 
    can be used for training or predictions using the scikit-learn library.  

    :param filename: path to the csv file with columns of features and the label
    :param selected_feature_cols: list of feature names, must be a subset of column names of the filename
                            if None --> select all features
    :param transformer: a fitted feature dataframe transformer for dimensionality reduction 
                        --> see get_pca_transformer-method for creation of such transformer.
    '''

    available_feature_cols = list(load_from_pickle(filename).set_index(INDEX_COL).iloc[:1].columns.values)
    available_feature_cols.remove(label_col)

    selected_feature_cols = list(selected_feature_cols) 
    selected_feature_cols.sort()

    assert(set(list(selected_feature_cols)).issubset(set(list(available_feature_cols))))

    data_df = load_from_pickle(filename)[selected_feature_cols + [INDEX_COL, LABEL_COL]].set_index(INDEX_COL)

    X_data, Y_labels = get_data_labels_from_df(data_df, selected_feature_cols, is_training, label_col = label_col, _balance_dataset = _balance_dataset, step_size = step_size, data_transformer = data_transformer, loc_indices = loc_indices)

    return X_data, Y_labels

def get_data_labels_from_df(data_df, selected_feature_cols, is_training, label_col = LABEL_COL, _balance_dataset = True, step_size = 1, data_transformer = None, loc_indices = None):
    
    
    iloc_indices = np.arange(len(data_df), step=step_size)

    if type(loc_indices) != type(None):
        iloc_indices = transform_df_loc_indices_iloc_accessible(data_df, loc_indices)

    X_data, Y_labels = data_df[selected_feature_cols], data_df[label_col]

    if type(data_transformer) != type(None):
        X_data = data_transformer.transform(X_data)
    else:
        X_data = np.array(X_data)

    # if is_training: 
    #     if _balance_dataset:
    #         X_data, Y_labels = balance_dataset(X_data, Y_labels)

    X_data = np.array(X_data)
    Y_labels = np.array(Y_labels, dtype = "int32")

    X_data = X_data[iloc_indices, :]
    Y_labels = Y_labels[iloc_indices]

    return X_data, Y_labels

def get_indices_from_file(filename):
    '''
    
    '''    
    data_df = load_from_pickle(filename).set_index(INDEX_COL)

    return data_df.index.values


def dataset_from_file(f,
                        signal_cols = SIGNAL_COLS,
                        label_col = LABEL_COL,
                        window_size = NN_CONFIG["WINDOW_SIZE"],
                        step_size = NN_CONFIG["STEP_SIZE"],
                        training = True,
                        batch_size = NN_CONFIG["BATCH_SIZE"],
                        activity_code_type_dict = ACTIVITY_CODE_TYPE_DICT_CHRIS,
                        dataset_range = (0, 1),
                        sampling_mode = "fixed",
                        loc_indices = None):

    '''
    Read dataframe from file and get a windowed dataset
    :param f: csv file
    :param window_size_samples: size of the window for the network
    :param dataset_range: 
    :param sampling_mode: see the tf dataset generator method
    :param loc_indices: overrules the adaptive or fixed sampling, these indices have to be the values from the index column
    '''
    try:
        f = f.decode()
        signal_cols = [s.decode() for s in signal_cols]
        label_col = label_col.decode()
    except:
        print("Decoding in load_file_and_process failed.")
        pass

    # Load the file as a dataframe and then create the datset
    if f.endswith(".csv"):
        data_df = pd.read_csv(f, index_col = INDEX_COL)
    elif f.endswith(".pkl"):
        data_df = load_from_pickle(f)
        data_df.set_index(INDEX_COL)
    else:
        file_ending = f.split(".")[-1]
        print(f"Invalid file ending {file_ending} in loading file. Supported formats are csv and pkl.")
        print()
        sys.exit()
    
    dataset = dataset_from_df(data_df,
                        signal_cols = signal_cols,
                        label_col = label_col,
                        window_size = window_size,
                        step_size = step_size,
                        training = training,
                        batch_size = batch_size,
                        activity_code_type_dict = activity_code_type_dict,
                        dataset_range = dataset_range,
                        sampling_mode = sampling_mode,
                        loc_indices = loc_indices)

    return dataset #, df_loc_indices

def transform_df_loc_indices_iloc_accessible(data_df, indices):
    '''
    Get the iloc accessible indices indices with which you could access
    data_df.iloc[df_iloc_indices] based on index values from data_df.loc
    Refer to the pandas documentation to see the difference, e.g. here:
    https://pandas.pydata.org/docs/user_guide/indexing.html 
    '''
    indices = np.array(indices).astype(int)    
    data_df["indices_iloc_accessible"] = np.arange(len(data_df))
    indices = [index for index in indices if index in data_df.index] # TODO investigate
    indices = data_df.loc[indices]["indices_iloc_accessible"].values
    return indices

def transform_df_iloc_indices_loc_accessible(data_df, df_iloc_indices):
    '''
    Get the loc indices i.e. those saved in data_df based on a list of 
    indices with which you could access data_df.iloc[df_iloc_indices]
    Refer to the pandas documentation to see the difference, e.g. here:
    https://pandas.pydata.org/docs/user_guide/indexing.html 
    '''
    df_iloc_indices = np.array(df_iloc_indices).astype(int)
    loc_indices = data_df.index.values[df_iloc_indices]
    return loc_indices

def get_loc_indices(data_df, step_size, window_size, sampling_mode, dataset_range = (0, 1), label_col = LABEL_COL, feature_file = False, label_distribution_dict = None):
    '''
    Based on sampling mode, step size, window size and datafram return loc indices 
    to sample from.

    #TODO adaptive sampling supply parameters

    :param sampling_mode: either "fixed" or "adaptive"
    :param label_distribution_dict: only for mode "adaptive", use this dictionary for calculating the class importances if submitted, otherwise use the indexed data
    '''
    # Split the activity with respect to their activity indices --> reduced amount of training time, especially less sitting task
    # The split only happens if the in the dataframe there is an ACTIVITY_IDX_COL with different values 
    # Therefore in the pipeline_config one has to set REDUCE_DATASET = True 
    # If there is no ACTIVITY_IDX_COL in the dataframe, the whole dataframe is considered one activity (manually an index of 0 is added)
    if not ACTIVITY_IDX_COL in data_df.columns.values:
        data_df[ACTIVITY_IDX_COL] = [0] * len(data_df)

    if sampling_mode == "adaptive":
        data_df = add_sampling_prob_col(data_df, step_size, window_size, feature_file = feature_file, activity_distribution_dict = label_distribution_dict) 

    loc_indices = []
    # Get the indices by iterating over the activities indicated by the ACTIVITY_IDX_COL
    for counter, act_index in enumerate(list(set(list(data_df[ACTIVITY_IDX_COL].values)))):
        data_df_activity_indexed = data_df.loc[data_df[ACTIVITY_IDX_COL] == act_index]
        if sampling_mode == "fixed":
            offset = window_size if (not feature_file) else 0
            loc_indices_indexed = list(data_df_activity_indexed.index.values[offset::step_size])
        elif sampling_mode == "adaptive": 
            loc_indices_indexed = list(get_adaptive_sampling_indices(data_df_activity_indexed, window_size, step_size, label_col = label_col, feature_file = feature_file))
        else:
            print(f"Invalid sampling mode {sampling_mode}. Valid modes are 'fixed' or 'adaptive'.")
            print(dataset_range)
            sys.exit()
        if len(loc_indices_indexed) == 0:
            continue
        
        # print(f"\n\n\ndataset range: {dataset_range[0]}, {dataset_range[1]}\n\n\n")
        if not ((dataset_range[0] == 0) & (dataset_range[1] == 1)):
            assert(dataset_range[0] < dataset_range[1])
            assert(dataset_range[0] >= 0)
            assert(dataset_range[1] <= 1)
            loc_indices_indexed = loc_indices_indexed[int(dataset_range[0] * len(loc_indices_indexed)):int(dataset_range[1] * len(loc_indices_indexed)) - 1]

        loc_indices.extend(loc_indices_indexed)
    
    return loc_indices

def get_labels_from_gt_loc_indices_files(gt_files, loc_indices_files_list):      
        
    assert(len(gt_files) == len(loc_indices_files_list))
    labels = []
    for gt_file, loc_indices_file in zip(gt_files, loc_indices_files_list):
            data = load_from_pickle(gt_file)
            indices = load_from_pickle(loc_indices_file)
            data_labels = list(data.loc[indices][LABEL_COL].values)

            labels.extend(data_labels) 

    return labels  

def get_labels_from_gt_files(gt_files):
    '''
    Return a list of all labels in ground truth files 
    Can be used to calculate label distributions 
    '''
    labels = []

    for gt_file in gt_files:
        data = load_from_pickle(gt_file)
        labels.extend(data[LABEL_COL].values) 

    return labels  


def get_loc_indices_list_from_files(data_files, step_size, window_size, sampling_mode, dataset_range = (0, 1), feature_file = False, loc_output_dir = None, undersample_on_label_distribution_shift = True, use_total_label_distribution = False):
    '''
    For a list of data files, load them and create loc indices at positions where to be sampled.

    :param feature_file: if feature_file == True --> no need for an offset 
    :param output_dir: where to save the loc files, if "", an output dir file be created in the parent directory of each file
    :param undersample_on_label_distribution_shift: previously underrepresented classes may be overrepresented (e.g. swing extension as it has a lot of transitions)
    :param use_total_label_distribution: for adaptive mode only. Whether or not to use the total label distribution (i.e. from all files) for assessing label importance

    :return: list of indices lists
    :return: furthermore, loc indices are saved in a temporary file so that the dataset generator can load these

    '''
    if len(data_files) == 0:
        print("Empty list for loc indices retrieval.")
        return [], []

    loc_indices_list = []   
    loc_indices_files = []
    out_dir_name = f"sampling_loc_indices_s{step_size}_w{window_size}_{sampling_mode}_({dataset_range[0]},{dataset_range[1]})"

    label_distribution_dict = None
    if sampling_mode == "adaptive":
        if use_total_label_distribution:
            all_labels =  get_labels_from_gt_files(data_files)
            label_distribution_dict = get_label_distr_dict_from_activity_list(all_labels)
        

    for f in data_files: 
        # Usually all data files are in the same directory
        # Save the loc indices in a pkl file
        if type(loc_output_dir) == type(None): 
            loc_output_dir_f = os.path.join(Path(f).parent, out_dir_name)
        else:
            loc_output_dir_f = loc_output_dir
        
        if not os.path.exists(loc_output_dir_f):
            os.makedirs(loc_output_dir_f)
        out_f = os.path.join(loc_output_dir_f, os.path.basename(f).replace(".pkl", "_sampling_indices.pkl"))
    
        if os.path.exists(out_f):
            loc_indices = load_from_pickle(out_f)
        else:
            f_labels = get_labels_from_gt_files([f])
            initial_distribution_dict = get_label_distr_dict_from_activity_list(f_labels)
            data_df = load_from_pickle(f)
            loc_indices = get_loc_indices(data_df, step_size, window_size, sampling_mode, dataset_range, feature_file = feature_file, label_distribution_dict = label_distribution_dict)
            
            if sampling_mode == "adaptive":
                if undersample_on_label_distribution_shift:
                    # avoid underrepresented classes to be overrepresented
                    loc_indices = undersample_indices_on_label_distribution_shift(data_df, loc_indices, initial_distribution_dict)
            
            save_as_pickle(loc_indices, out_f)
        
        loc_indices_list.append(loc_indices)
        loc_indices_files.append(out_f)

    return loc_indices_list, loc_indices_files

def dataset_from_df(data_df,
                        signal_cols = SIGNAL_COLS,
                        label_col = LABEL_COL,
                        window_size = NN_CONFIG["WINDOW_SIZE"],
                        step_size = NN_CONFIG["STEP_SIZE"],
                        training = False,
                        batch_size = NN_CONFIG["BATCH_SIZE"],
                        activity_code_type_dict = ACTIVITY_CODE_TYPE_DICT_CHRIS,
                        dataset_range = (0, 1),
                        sampling_mode = "fixed",
                        loc_indices = None):
    '''
    :param loc_indices: the indices with which you access dataframe.loc[indices]
                    if submitted as list or array this will overrule the dataset_range and sampling mode chosen! 
                    and not dataframe.iloc[indices]
                    ! caution !
                    the indices will be transformed in a way they can be accessed via iloc but if you deliver 
                    indices here they mean the indices from the data_df.index method.
    :param return: dataset, df_loc_indices
                dataset object over which you can iterate and that holds the data and label information
                df_loc_indices: the index values of the dataframe corresponding to every dataset entry 
    '''

    # if the loc_indices parameter is submitted --> take these indices 
    if type(loc_indices) == type(None):        
        print("Getting loc indices in dataset_from_df - method.")
        loc_indices = get_loc_indices(data_df, step_size, window_size, sampling_mode, dataset_range)
    else:
        print("Using submitted loc indices in datset_from-df - method.")
            
    if training: 
        random.Random(23).shuffle(loc_indices)
    
    iloc_indices = transform_df_loc_indices_iloc_accessible(data_df, loc_indices)
    assert(np.min(iloc_indices) - window_size >= 0)
    assert(np.max(iloc_indices) < len(data_df))

    dataset = get_windowed_tf_dataset_from_indices(data_df,
        signal_cols=signal_cols,
        label_col = label_col,
        window_size_samples = window_size,
        batch_size = batch_size,
        end_indices = iloc_indices,
        activity_code_type_dict = activity_code_type_dict)
    
    return dataset

def undersample_indices_on_label_distribution_shift(data_df, loc_indices, label_distribution_dict_initial, label_col = LABEL_COL):
    '''
    undersample classes that after applying adaptive sampling now have more samples than the original

    :param data-df: 
    :param loc_indices: the indices returned from the adaptive sampling algorithm
    :param label_distribution_dict_initial: the initial label distribution before applying adaptive sampling

    '''
    label_distribution_dict_out = get_label_distr_dict_from_activity_list(data_df.loc[loc_indices][label_col].values)
    
    initial_keys = list(label_distribution_dict_initial.keys())
    initial_keys = [key for key in initial_keys if key in label_distribution_dict_out.keys()]
    initial_vals = [label_distribution_dict_initial[k] for k in initial_keys]  
  
    initial_vals, initial_keys  = zip(*sorted(zip(initial_vals, initial_keys), reverse=True))
    initial_vals, initial_keys  = list(initial_vals), list(initial_keys)

    while True: 
        undersampled = False
        label_distribution_dict_out = get_label_distr_dict_from_activity_list(data_df.loc[loc_indices][label_col].values)

        for k_prev, k_post, in zip(initial_keys[:-1],initial_keys[1:]):

            if label_distribution_dict_out[k_post] > label_distribution_dict_out[k_prev]:
                 
                num_items_to_remove = label_distribution_dict_out[k_post] - label_distribution_dict_out[k_prev]

                data_df = data_df.loc[loc_indices]
                loc_indices_to_undersample = data_df.loc[data_df[label_col] == k_post].index.values

                loc_indices_to_remove = random.sample(list(loc_indices_to_undersample), num_items_to_remove)

                loc_indices = [l for l in loc_indices if l not in loc_indices_to_remove]

                undersampled = True
                break

        if not undersampled:
            return loc_indices

def tf_dataset_generator(file_list,
                    signal_cols,
                    label_col,
                    window_size_samples,
                    window_step_size,
                    shuffle,
                    batch_size = NN_CONFIG["BATCH_SIZE"],
                    activity_code_type_codes = list(ACTIVITY_CODE_TYPE_DICT_CHRIS.keys()),
                    activity_code_type_types = list(ACTIVITY_CODE_TYPE_DICT_CHRIS.values()),
                    dataset_range = (0, 1), 
                    sampling_mode = "fixed",
                    loc_indices_files_list = [],
                    print_input_files_information = False):
    '''
    The data generator. Iterate over the file list and returns data chunks with size of batch size
    :param balance_dataset: #TODO
    :param range: by default (0, 1) gives the range of data to be used 
                e.g. for training and val dataset generation you could use (0, 0.7) and (0.7, 1.0)

    if you want to debug: 
    gen = tf_data_generator_indexed(*args)
    for x in gen:
        print("")
    
    :param file_list: path to the files containing the data
    :param sampling_mode:   "fixed"     --> us a fixed step size for shifting the window
                            "adaptive"  --> increase sampling rate in areas of activity changes
    :param loc_indices_list: overrules fixed or adaptive mode
                each entry describes the indices list from which to load the samples

    '''
    try: 
        loc_indices_files_list = [f.decode() for f in loc_indices_files_list]
    except:
        loc_indices_files_list = loc_indices_files_list

    if len(loc_indices_files_list) != len(file_list):
        loc_indices_list = [None] * len(file_list) 
        print("Getting loc indices during generation.")
    else:
        loc_indices_list = [load_from_pickle(f) for f in loc_indices_files_list]

    # The tf data generator called via tf.data.Dataset.from_generator cannot handle python dictionaries
    activity_code_type_dict = {}
    for code_, type_ in zip(activity_code_type_codes, activity_code_type_types):
        try:
            type_ = type_.decode()
            type_ = type_.replace("b", "")
            type_ = type_.replace("'", "")
        except:
            type_ = type_
        activity_code_type_dict[code_] = type_

    try: 
        sampling_mode = sampling_mode.decode()
    except:
        sampling_mode = sampling_mode

    # Shuffles the file list every iteration 
    tmp = list(zip(file_list, loc_indices_list))
    random.shuffle(tmp)
    file_list, loc_indices_list = zip(*tmp)

    if print_input_files_information:
        print(f"\ntf_data_generator got following input files (shuffle = {shuffle}, window size = {window_size_samples}, window step size = {window_step_size}):")
        for file in file_list:
            print(os.path.basename(file))
        print("\n")

    for i, (file, loc_indices) in enumerate(zip(file_list, loc_indices_list)):
        dataset = dataset_from_file(file, signal_cols = signal_cols, label_col = label_col, window_size = window_size_samples, step_size = window_step_size, training = shuffle, batch_size = batch_size, activity_code_type_dict = activity_code_type_dict, dataset_range = dataset_range, sampling_mode = sampling_mode, loc_indices = loc_indices)

        for x, y in dataset:
            yield np.asarray(x), np.asarray(y)

def get_label_encoder(labels = list(ACTIVITY_CODE_TYPE_DICT_CHRIS.keys())):
    '''
    For the xgboost model labels from 0, 1, 2, 3, ... are necessary
    Using a label encoder our class labels 0, 100, 200, ... are converted
    This encoder then also can be used to inverse the encoding
    '''
    labels.sort()

    le = LabelEncoder()
    le.fit(labels)

    return le

def encode_classweight_dict(classweight_dict, activity_dict = ACTIVITY_CODE_TYPE_DICT_CHRIS):

    initial_keys = list(activity_dict.keys())
    
    le = get_label_encoder(labels = initial_keys)

    transformed_classweight_dict = {}

    for initial_key in initial_keys:
        transformed_classweight_dict[le.transform([initial_key])[0]] = classweight_dict[initial_key]

    return transformed_classweight_dict
