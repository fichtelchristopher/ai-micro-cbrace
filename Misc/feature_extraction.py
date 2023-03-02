''' 
Master thesis project of Christopher Fichtel started in Oct 2022
AI for microprocessor-controlled knee joints

File created 16 Nov 2022

Code base for windowed feature generation (for the feature based AI approach)
'''

from Configs.namespaces import *

import matplotlib.pyplot as plt

from Configs.pipeline_config import *


from Visualisation.visualisation import *

def add_feature_col(data_df, signal_name, window_size_samples = 50,  metric = "mean", feature_col_out = None):
    '''

    '''
    if type(feature_col_out) is type(None): 
        feature_col_out = get_feature_col_out_name(signal_name, window_size_samples, metric)

    if metric == "mean":
        data_df[feature_col_out] = data_df[signal_name].rolling(window = window_size_samples).mean()
    elif metric == "std":
        data_df[feature_col_out] = data_df[signal_name].rolling(window = window_size_samples).std()
    elif metric == "min":
        data_df[feature_col_out] = data_df[signal_name].rolling(window = window_size_samples).min()
    elif metric == "max":
        data_df[feature_col_out] = data_df[signal_name].rolling(window = window_size_samples).max()
    elif metric == "range":
        data_df[feature_col_out] = data_df[signal_name].rolling(window = window_size_samples).apply(get_window_range)
    elif metric == "init":
        data_df[feature_col_out] = data_df[signal_name].rolling(window = window_size_samples).apply(get_window_initial_value)
    elif metric == "final":
        data_df[feature_col_out] = data_df[signal_name].rolling(window = window_size_samples).apply(get_window_final_value)
    else:
        print(f"Unknown metric {metric}. Returning dataframe without calculating features.")

    # plt.plot(data_df[signal_name])
    # plt.plot(data_df[feature_col_out])
    # plt.clf()

    return data_df

def get_window_range(window):
    '''
    '''
    return np.abs(np.max(window)-np.min(window))

def get_window_initial_value(window):
    '''
    Custom Rolling function for the "init" window metric
    '''
    return window.values[0]

def get_window_final_value(window):
    '''
    Custom Rolling function for the "final" window metric
    '''
    return window.values[-1]

def get_feature_col_out_name(signal_name, window_size_samples, metric):
    '''
    
    :param feature_name: the name of the feature column
    :param window_size: the window size in samples
    :param metric: one of "mean", "std", "min", "max", "init", "final"
        "init" (first value in the window)
        "final" (last value in the window)
    '''
    if metric == "final":
        return f"{signal_name}_{metric}" 
    else:
        return f"{signal_name}_{window_size_samples}_{metric}"


def feature_generation(data_df, signal_names, window_size_samples, metrics, existing_feature_cols = None):
    '''
    For each sensor signal (signal names) --> six features namely "mean", "std", "min", "max", "init" (first value in the window), "final" (last value of the window) 

    :param data_df: input dataframe
    :param signal_names: list the signal/column names for which features should be extracted
            signal_names must be subset of the dataframe's column names
    :param window_size_samples: 
    :param existing_feature_cols: 
        if None --> create all the features
        otherwise list of existing names: if the feature_col_out_name is in this list, skip this column --> avoids unnecessary computation

    '''
    output_feature_cols = []

    for s in signal_names:
        # signal_cols = [s]

        for metric in metrics: 
            feature_col_out_name = get_feature_col_out_name(s, window_size_samples, metric)

            if type(existing_feature_cols) != type(None):

                if feature_col_out_name in existing_feature_cols:
                    # skip this feature, it already exists
                    continue

            data_df = add_feature_col(data_df, s, metric = metric, window_size_samples = window_size_samples, feature_col_out = feature_col_out_name)
            output_feature_cols.append(feature_col_out_name)
            # signal_cols.append(feature_col_out_name)

        # plot_signals_from_df(data_df, data_cols=signal_cols)
        # plt.legend()
        # plt.show()
        # plt.clf()

    return data_df, output_feature_cols

    # add windows mean col


def get_output_feature_cols(signal_names, window_sizes_samples, metrics): 
    '''
    :param signal_names: list of input column signals for which features are generated
    :param window_sizes_samples: list of window sizes used for feature generation, if only one window size is used
                                deploy as [window_size]
    :param metrics: list of metrics for which feature is computed 
    '''
    assert(len(signal_names) > 0)
    assert(len(window_sizes_samples) > 0)
    assert(len(metrics) > 0)

    output_feature_cols = []

    for w in window_sizes_samples:
        for s in signal_names:
            for metric in metrics: 
                feature_col_out_name = get_feature_col_out_name(s, w, metric)
                output_feature_cols.append(feature_col_out_name)

    return output_feature_cols
    