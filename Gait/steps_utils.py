''' 
Master thesis project of Christopher Fichtel started in Oct 2022
AI in operating ortheses

File created 11 Jan 2023

Useful functions for generating typical steps. Methods mainly moved from Misc/utils.py
'''


import numpy as np
import scipy.interpolate
from Misc.math import *
import os
from Configs.namespaces import *
from Processing.loading_saving import *
from Processing.preprocessing import *
from Configs.pipeline_config import * 
from Misc.math import * 
from Visualisation.visualisation import *
from Visualisation.namespaces_visualisation import *
from Misc.utils import *

import matplotlib
matplotlib.use("TkAgg")

def extract_steps(aos_gt_df, activity_dict_walking_classes):
    '''
    *** for aos gt files ***

    :param aos_gt_df: the ground truth dataframe generated with generated with ground_truth_generation.py
    :param walking_code_type_dict: a dictionary containing only the walking classes
                            e.g. {200: "walking-no-flexion",
                                  300: "walking-w-flexion",
                                  400: "yielding"}
                            the integer codes correspond to the LABEL_COL in the aos_gt_df
    :return: step_data_dict, dictionary, per walking class one dictionary entry
                    key: is the walking class code int (e.g. 200)
                    item: list of dataframes that describe the step, i.e. the data between two heel strikes,
                        one list entry thus corresponds to one step
    '''
    
    existing_walking_activity_codes = list(activity_dict_walking_classes.keys())

    # initialize the output directory
    step_data_dict = {}
    for code in existing_walking_activity_codes:
        step_data_dict[code] = []

    ic_indices = aos_gt_df[aos_gt_df[IC_COL] == 1].index.values

    counter = 0

    for (step_start, step_stop) in zip(ic_indices[:-1], ic_indices[1:]):
        step_df = aos_gt_df[(step_start < aos_gt_df.index) & (aos_gt_df.index < step_stop)]
        # Check if one of the walking classes is "active" between the two initial contacts
        # if counter == 200:
        #     break

        for code in existing_walking_activity_codes:
            if code in step_df[LABEL_COL].values: 

                counter += 1
                

                # Investigation
                # num_samples_pre_post = 100
                # joint_angle = list(step_df["JOINT_ANGLE"].values)
                # joint_angle_max_idx = joint_angle.index(max(joint_angle))
                # if (code == 200) & (max(joint_angle) > 30):
                #     if joint_angle_max_idx < 0.4 * len(step_df):
                #         step_pre_post = aos_gt_df[(step_start - num_samples_pre_post < aos_gt_df.index) & (aos_gt_df.index < step_stop + num_samples_pre_post)]
                #         plt.clf()
                #         plot_signals_from_df(step_pre_post, data_cols = ["JOINT_ANGLE", "DDD_ACC_TOTAL", "RI_RULID1", "JOINT_ANGLE_VELOCITY"], data_labels = ["ANGLE", "ACC-TOTAL", "RI_RULID1", "JOINT_ANGLE_VELOCITY"], data_scaling_factors = [1, 1, 1, 1 / 10])
                #         plot_indices([num_samples_pre_post, len(step_pre_post)-num_samples_pre_post], color = SIGNAL_COLOR_DICT[IC_COL], alpha = 1.0, ymax = 20, ymin = 0, label = IC_COL)
                #         plt.show()

                step_data_dict[code].append(step_df)
                continue
    
    return step_data_dict


def countOccurrence(a):
  k = {}
  for j in a:
    if j in k:
      k[j] +=1
    else:
      k[j] =1
  return k


def get_subject_step_transitions_from_activity_progression(activity_progression, output_file = None, activity_code_type_dict = None, discard_swext = True):
    '''
    :param activity_progression: the ground truth activity as list or numpy array

    :param discard_swext:
        True --> swing extension class will be discarded by assigning it the label of the previous activity 
                i.e. a level walking step [LW, LW, LW, SWEXT, SWEXT] will then be [LW, LW, LW, LW, LW]
                requires that an activity_code_type_dict is submitted and SWING_EXTENSION_CLASSNAME is in the dictionary's values
    '''
    transition_df = pd.DataFrame(columns=["from", "to", "num"])

    if type(activity_code_type_dict) == type(None):
        # simply use the labels from the activity progression
        activity_code_type_dict = {}
        available_labels = list(set(list(activity_progression)))

        for l in available_labels: 
            activity_code_type_dict[l] = l

    compressed_activity_progression, start_indices = compress_state_progression(activity_progression)

    if discard_swext:
        assert(SWING_EXTENSION_CLASSNAME in list(activity_code_type_dict.values()))
        activity_type_code_dict = swap_key_entry_dict(activity_code_type_dict)
        swing_ext_code = activity_type_code_dict[SWING_EXTENSION_CLASSNAME]
        compressed_activity_progression = [a for a in compressed_activity_progression if a != swing_ext_code]
    
    trans_from  = np.array(compressed_activity_progression[:-1])
    trans_to    = np.array(compressed_activity_progression[1:])

    transitions = [(f, t) for f, t in zip(trans_from, trans_to)]

    transition_occurences_dict = countOccurrence(transitions)

    for transition, val in transition_occurences_dict.items():
        transition_df = transition_df.append({"from": activity_code_type_dict[transition[0]], "to": activity_code_type_dict[transition[1]], "num": val}, ignore_index = True)

    transition_dict = {}
    for index, row in transition_df.iterrows():
        transition_dict[str(row["from"]) + "-->" + str(row["to"])] = str(row["num"])

    if type(output_file) != type(None):
        transition_df.to_csv(output_file, index = False)

        visualize_dict_as_boxplot(transition_dict, output_file.replace(".csv", ".png"))

    return transition_dict, transition_df

def get_subject_step_transitions_dict_from_file(subject_file, output_dir, activity_code_type_dict = ACTIVITY_CODE_TYPE_DICT_CHRIS):
    '''
    Get all possible transitions and their distribution

    includes transitions from "other" "walking" and "yielding" classes

    '''
    assert(subject_file.endswith(".pkl"))

    gt_df = load_from_pickle(subject_file)

    output_file = os.path.join(output_dir, os.path.basename(subject_file).replace(".pkl", "_transitions.csv"))

    transition_dict = get_subject_step_transitions_dict_from_df(gt_df, output_file, activity_code_type_dict)

    return transition_dict

def get_subject_step_transitions_dict_from_df(gt_df, output_file = None, activity_code_type_dict = ACTIVITY_CODE_TYPE_DICT_CHRIS):
    '''
    see get_subject_step_transitions_from_file()
    '''
    activity_progression = gt_df["ACTIVITY"].values
    
    transition_dict, transition_df = get_subject_step_transitions_from_activity_progression(activity_progression, output_file, activity_code_type_dict = activity_code_type_dict)

    return transition_dict

def create_subject_typcial_steps_aos(subject_file, signals: list, create_plot = True, output_dir = None, activity_code_type_dict = ACTIVITY_CODE_TYPE_DICT_CHRIS, save_typical_steps_pkl = True): 
    '''
    For aos files only. Does the same as the create_subject_typical_steps_leipzig(). However, leipzig expects one file per step label.
    In an aos gt file a step is defined in the range between two initial contact indices and the class label in between. 

    :param subject_files:   path to a ground truth file of aos data. A ground truth file can be generated with 
                            the ground_truth_generation.py script. Refer to this file for more information
                            and thorough explanation. 
                            Ends on .pkl

    :param signals: list of column names (str) for which the typical plot should be created

    :param create_plot: whether or not to plot tyapical steps
    :param subject_output_dir: if submitted --> save plots in this directory

    :return:    1. dictionary of typical steps; key is the step label and the value is a dataframe containing for each signal a column f"{signal}_mean" and f"{signal}_std"
                2. dictionary of interpolated steps: key is the step label and value a list of dataframes containing for each signal a column with interpolated values
                3. signals - the used signals
    
    '''
    if save_typical_steps_pkl: 
        assert(type(output_dir) != type(None))
        typical_steps_output_f = os.path.join(output_dir, os.path.basename(subject_file).replace(".pkl", "_typical_steps_dict.pkl"))
        if os.path.exists(typical_steps_output_f): 
            typical_step_dfs_dict = load_from_pickle(typical_steps_output_f)
            return typical_step_dfs_dict

    cols = signals + [IC_COL, SWR_COL, LABEL_COL, INDEX_COL]

    # disgard sitting and other class
    activity_dict_walking_classes = {}
    for walking_code, walking_type in activity_code_type_dict.items():
        if ("walking" in walking_type) or ("yielding" in walking_type):
            activity_dict_walking_classes[walking_code] = walking_type 

    if subject_file.endswith(".csv"):
        try:
            aos_gt_df = pd.read_csv(subject_file, usecols = cols, index_col = INDEX_COL)
        except:
            aos_gt_df = pd.read_csv(subject_file, usecols = cols, index_col = INDEX_COL)
    elif subject_file.endswith(".pkl"):
        aos_gt_df = load_from_pickle(subject_file)
    else:
        print(f"Invalid file ending for {subject_file} in create_subject_typcial_steps_aos method.")
        sys.exit()

    step_data_dict =  extract_steps(aos_gt_df, activity_dict_walking_classes)

    # typical and interpolated step data dictionaries initialized
    typical_step_dfs_dict, interpolated_step_dfs_dict, average_durations_dict = get_typical_interpolated_dfs_dicts(step_data_dict, signals)

    # Create the plots
    if create_plot:
        for walking_code in typical_step_dfs_dict.keys():
            if output_dir:
                output_file = os.path.join(output_dir, f"{activity_dict_walking_classes[walking_code]}_{len(signals)}signals.png")
            else:
                output_file = None
            plot_typical_steps( typical_step_dfs_dict[walking_code],
                                signals,
                                interpolated_step_dfs = interpolated_step_dfs_dict[walking_code],
                                title = f"{len(step_data_dict[walking_code])} {activity_dict_walking_classes[walking_code]} steps\n Duration (samples): {average_durations_dict[walking_code][0]}$\pm${ average_durations_dict[walking_code][1]}",
                                output_file=output_file)

    if save_typical_steps_pkl:
        save_as_pickle(typical_step_dfs_dict, typical_steps_output_f)

    return typical_step_dfs_dict



def create_subject_typical_steps_leipzig(subject_files, signals: list, create_plot = True, output_dir = None):
    '''
    For Leipzig files only.

    :param subject_files:   list, paths to txt files of leipzig data for one subject. These data files should be in the respective "labeled" data folder 
                            of the subject and contain .mat-files with indices indicating step ranges
                            One file here represents a label
    :param signals: list of column names (str) for which the typical plot should be created

    :param create_plot: whether or not to plot tyapical steps
    :param subject_output_dir: if submitted --> save plots in this directory

    :return:    1. dictionary of typical steps; key is the step label and the value is a dataframe containing for each signal a column f"{signal}_mean" and f"{signal}_std"
                2. dictionary of interpolated steps: key is the step label and value a list of dataframes containing for each signal a column with interpolated values
                3. signals - the used signals
    
    '''
    if output_dir:
        assert(os.path.isdir(output_dir) & os.path.exists(output_dir))

    # per file find the corresponding label files and create the step-label indices dictionary (see respective functions for explanaitions)
    labels_indices_dicts = [get_labels_indices_dict(get_leipzig_label_files(f)) for f in subject_files]

    # Get a list of all available steps
    available_steps = []
    for dict in labels_indices_dicts:
        available_steps += dict.keys()
    available_steps = list(set(available_steps))

    # Load the data frames and preprocess
    data_dfs = [load_data_file(f) for f in subject_files]
    data_dfs = [preprocess_df(df, knee_lever = KNEE_LEVER) for df in data_dfs]

    # Initialize the dictionary that for each step will hold a list of "step dataframes"
    step_data_dict = {}
    for step in available_steps: 
        step_data_dict[step] = []  

    # Exctract individual step dataframes and add them to the respective entry in step_data_dict
    for data_df, labels_indices_dict in zip(data_dfs, labels_indices_dicts):
        for step_label, step_indices_list in labels_indices_dict.items():
            for step_indices in step_indices_list:
                step_data_dict[step_label].append(data_df.loc[step_indices[0]:step_indices[1], signals])
    
    # typical and interpolated step data dictionaries initialized
    typical_step_dfs_dict, interpolated_step_dfs_dict, durations_dict = get_typical_interpolated_dfs_dicts(step_data_dict, signals)

    if create_plot:
        for step_label in typical_step_dfs_dict.keys():
            if output_dir:
                output_file = os.path.join(output_dir, f"{step_label}_{len(signals)}.png")
            else:
                output_file = None
            plot_typical_steps(typical_step_dfs_dict[step_label], signals, interpolated_step_dfs = interpolated_step_dfs_dict[step_label], title = STEP_LABEL_DICT_LEIPZIG[step_label], output_file=output_file)

    return typical_step_dfs_dict, interpolated_step_dfs_dict, signals

def get_typical_interpolated_dfs_dicts(step_data_dict: dict, signals: list):
    '''
    :step_data_dict:    dictionary
                        step labels as key and a list of dataframes as values, each dataframe represents a step
                        each signal is a column in the dataframe
    :param signals: list of column names (str) for which the typical plot should be created
    '''
    typical_step_dfs_dict       = {}
    interpolated_step_dfs_dict  = {}
    durations_dict      = {}

    for step_label, step_dfs in step_data_dict.items():
        # step_dfs is a list of dataframes

        interpolated_step_dfs = [get_interpolated_df(step_df) for step_df in step_dfs]
        typical_step_df = pd.DataFrame()

        for signal in signals: 

            typical_step_mean, typical_step_std = get_typical_step_mean(interpolated_step_dfs, signal)
            typical_step_df[signal + TYPICAL_STEP_MEAN_STR] = typical_step_mean
            typical_step_df[signal + TYPICAL_STEP_STD_STR] = typical_step_std
        
        duration_samples_mean, duration_samples_std = get_typical_step_length_mean_std(step_dfs)

        interpolated_step_dfs_dict[step_label]  = interpolated_step_dfs
        typical_step_dfs_dict[step_label]       = typical_step_df
        durations_dict[step_label]      = (duration_samples_mean, duration_samples_std)

    return typical_step_dfs_dict, interpolated_step_dfs_dict, durations_dict


def create_inter_subject_typical_steps(typical_step_dfs_dict_list, signals, create_plot = True, output_dir = None, mode = "aos"):
    '''
    Create inter subjec typical step. Of multiple subjects take the typical step (i.e. the mean) and created a 
    inter subject mean and standard variation for a typical step.
    
    :param typical_step_dfs_dict_list:  list of typical_step_dfs_dict (returned by create_subject_typical_steps_leipzig, look at this method for explanation)
                                        one entry per subject
    :param mode: "aos" for aos data or "leipzig" for Leipzig data
    :return: typical_step_df_inter_subject_dict, per 
    '''
    signals = [s + TYPICAL_STEP_MEAN_STR if (TYPICAL_STEP_MEAN_STR not in s) else s for s in signals]

    # Prepare a dictionary with step label as key and list of dataframes as entries
    # In order to be able to call get_typical_interpolated_dfs_dict()
    step_data_dict_out = {}
    for step_data_dict_subject in typical_step_dfs_dict_list:
        for step_label, df in step_data_dict_subject.items():
            if step_label not in step_data_dict_out.keys():
                step_data_dict_out[step_label] = [df]
            else: 
                step_data_dict_out[step_label] = step_data_dict_out[step_label] + [df]

    typical_step_dfs_dict, interpolated_step_dfs_dict, durations_dict = get_typical_interpolated_dfs_dicts(step_data_dict_out, signals)

    if create_plot:
        for step_label in typical_step_dfs_dict.keys():
            if output_dir:
                output_file = os.path.join(output_dir, f"{step_label}_{len(signals)}.png")
            else:
                output_file = None

            step_label_dict = STEP_LABEL_DICT_LEIPZIG if mode == "leipzig" else ACTIVITY_CODE_TYPE_DICT_CHRIS

            plot_typical_steps(typical_step_dfs_dict[step_label], signals, interpolated_step_dfs = interpolated_step_dfs_dict[step_label], title = step_label_dict[step_label], output_file=output_file)

    return typical_step_dfs_dict


def get_steps_interpolated(step_dfs, num_values = 101):
    '''
    Create a typical step defined by mean and std.s

    :param step_dfs:    list of step dataframes of a step type
    
    :return: list of step arrays of the typical step in percentage of gait cycle --> 100 steps
    '''
    
    interpolated_step_dfs = [get_interpolated_df(step_df, num_values=num_values) for step_df in step_dfs]
    return interpolated_step_dfs


def get_typical_step_mean(step_dfs: list, signal):
    '''
    Create a typical step defined by mean and std.

    :param step_dfs:    list of step dataframes of a step type (e.g. "RA"), i.e. this list only contains steps from the same label! 
    :param signal:      signal for which the typical step should be created, must be in the column values of a dataframe
    
    :return: array of the typical step in percentage of gait cycle --> 100 steps
    '''
    interpolated_step_dfs = get_steps_interpolated(step_dfs, num_values=101)

    interpolated_signal_steps = []
    for interpolated_step_df in interpolated_step_dfs:
        interpolated_signal_steps.append(interpolated_step_df[signal])
    interpolated_signal_steps = np.array(interpolated_signal_steps)  # shape is (number of found steps, 101)

    typical_step_mean   = np.mean(interpolated_signal_steps, axis = 0)
    typical_step_std    = np.std(interpolated_signal_steps, axis = 0)
    return typical_step_mean, typical_step_std

def get_typical_step_length_mean_std(step_dfs: list):
    '''
    :param step_dfs: list of pandas dataframes. Each dataframe corresponds to the length of one step 
    '''
    lengths = [len(step_df) for step_df in step_dfs]

    if len(lengths) == 0:
        return 0, 0

    return int(np.mean(lengths)), int(np.std(lengths))

def extract_swr_from_df(step_df):
    '''
    :param step_df: the dataframe describing one step

    :return: the position of the swing phase reversal
    '''

    return x 