''' 
Master thesis project of Christopher Fichtel started in Oct 2022
AI in operating ortheses

File created 06 Oct 2022

Code Base for general utils
'''

import numpy as np

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import scipy
import scipy.interpolate
from .math import *
import os
from Configs.namespaces import *
from Processing.loading_saving import *
from Processing.preprocessing import *
from Configs.pipeline_config import * 
from Misc.math import * 
from Visualisation.visualisation import *
from Visualisation.namespaces_visualisation import *
import sys
import psutil
import platform
from datetime import datetime
from pathlib import Path

import pickle as pkl

import shutil

def get_absolute_acceleration(df):
    '''
    :param df: dataframe containing the columns "DDD_ACC_X", "DD_ACC_Y", "DDD_ACC_Z"
    :return: absolute acceleration as numpy array (same length as df)    
    '''
    assert set(ACC_COLS).issubset(set(list(df.columns.values)))

    df_acc =  np.array(df[ACC_COLS].values)

    acc_abs = rsm(df_acc)

    return acc_abs

def get_acceleration_df(df):
    '''
    :param df: dataframe containing the columns "DDD_ACC_X", "DD_ACC_Y", "DDD_ACC_Z"
    :return: dataframe containing only the columns for accelerations  
    '''
    assert set(ACC_COLS).issubset(set(list(df.columns.values)))

    return df[ACC_COLS]

def get_df_duration_in_seconds(df, start_stop_index = None):
    '''
    Computation of dataframes duration based on sampling frequency values.

    :param df: dataframe that has to contain the SAMPLING_FREQ_COL from which the duration is derived  
    :param start_stop_index:    optional range for which duration should be found out 
                                supply as tuple like: (start_idx, stop_idx)
    '''
    if type(start_stop_index) == type(None):
        start_stop_index = (0, -1)

    sampling_freq = df[SAMPLING_FREQ_COL].iloc[start_stop_index[0]:start_stop_index[1]]

    duration = (1/sampling_freq).sum()

    return duration

def get_set_rule_combinations(df, output_file = None):
    '''
    Return a table with existing rule - set combinations in the underlying dataframe
    
    :param df: input dataframe containing the set/rule id columns ("RI_SETIDX", "RI_RULIDX" with X being the corresponding integer)
    :output_file: 
        = None (default): no saving of results 
        = str - filepath to textfile: save set rule combinations as text in file but only if the file does not exist at the moment of calling this method
    :return: list of unique set-rule combinations, datatype of list entry is a pandas dataframe
    '''
    set_cols = ["RI_SETID0", "RI_SETID1", "RI_SETID2", "RI_SETID3"]
    rule_cols = ["RI_RULID0", "RI_RULID1", "RI_RULID2", "RI_RULID3"]

    set_rule_id_combinations = []
    for set_col, rule_col in zip(set_cols, rule_cols):
        set_rule_comb = df.groupby([set_col, rule_col]).size().reset_index().rename(columns={0:'count'})
        set_rule_comb.reset_index(drop = True, inplace = True)
        set_rule_comb = set_rule_comb.astype(int)
        set_rule_id_combinations.append(set_rule_comb)
    
    if output_file: 
        assert output_file.split(".")[-1] == "txt"
        if not os.path.exists(output_file):
            with open(output_file, "a") as f:
                for set_rule_comb in set_rule_id_combinations:
                    print(set_rule_comb, file = f)
                    print("\n\n\n", file = f)

    return set_rule_id_combinations

def get_acc_batch_var(data_df, batch_size = 1000):
    '''
    Return batch acceleration variance. 
    '''
    df_acc = get_acceleration_df(data_df)

    acc_var = get_batchwise_var(df_acc, batch_size)

    return acc_var


def get_reversed_dictionary(activity_type_code_dict = ACTIVITY_TYPE_CODE_DICT_CHRIS):
    '''
    activity_code_type_dict is a dictionary that assigns each name (e.g. walking) an integer

    :return: a dictionary where key and item are swithced 
    '''
    activity_code_type_dict = {} # "reverse" dictionary with the integer values as keys
    for key, val in activity_type_code_dict.items():    
        activity_code_type_dict[val] = key 
    
    return activity_code_type_dict

def get_ruleset_type_from_f(filepath): 
    '''
    Based on the data in a file determine whether the "product" or "study"
    For an explanation of these terms also refer to the RULE_SET_LIBRARIES variable in the namespaces.py file

    :param filepath: path to the mat or csv file you want to analyze
    :return: the used library - "study" or "product"
    '''
    assert(os.path.exists(filepath))
    assert(filepath.split(".")[-1] in ["csv", "mat"]) # check that the file is a csv or mat file

    aos_data_df = load_data_file(filepath)

    return get_ruleset_type_from_df(aos_data_df) 


def get_ruleset_type_from_df(df):
    '''
    Same as get_rule_set_library_from_f() only that the method parameter is a df 
    Use this method if the dataframe has already been loaded via the load_data_file()-method

    So far the rule-set library will be determined based on the set-rule id combinations for .... # TODO 

    :param df: dataframe created with the load_data_file()-method
    :return: the used library - "study" or "product"
    '''

    set_rule_id_combinations = get_set_rule_combinations(df)

    id_0_combinations = set_rule_id_combinations[0]
    id_1_combinations = set_rule_id_combinations[1]
    id_2_combinations = set_rule_id_combinations[2]
    id_3_combinations = set_rule_id_combinations[3]
    
    id_1_set_4 = id_1_combinations.loc[id_1_combinations["RI_SETID1"] == 4] # where RI_SETID1 == 4
    id_1_set_6 = id_1_combinations.loc[id_1_combinations["RI_SETID1"] == 6] # where RI_SETID1 == 6

    if id_1_set_4.empty:
        return "product"

    if id_1_set_6.empty:
        return "study"

    if len(id_1_set_6) / len(id_1_set_4) > 0.1:
        return "product"

    if len(id_1_set_4) / len(id_1_set_6) > 0.1:
        return "study"

    return None # need other method, e.g. include joint angle #TODO


def get_start_stop_indices(activity_indices): 
    '''
    :activity_indices: list or array of indices where a certain activity is e.g. [1,2,3,8,9,10]
    :return: list of start indices and stop indices of certain activity     e.g. [1, 8], [3, 10]
    '''

    activity_indices = list(activity_indices)
    activity_indices = [-999] + activity_indices # to make sure the first index is considered a start index
    start_indices = [i_nex for (i_prev, i_nex) in zip(activity_indices[:-1], activity_indices[1:]) if (i_nex > i_prev + 1) ]
    stop_indices =  [i_prev for (i_prev, i_nex) in zip(activity_indices[1:-1], activity_indices[2:]) if (i_nex > i_prev + 1) ] + [activity_indices[-1]]

    return start_indices, stop_indices

def get_labels_indices_dict(label_files):
    '''
    :param label_files: list of paths to Leipzig label files. Leipzig label file looks like this: 2018-05-16_15-43-05-label_RA.mat where RA is the label.
    :return: dictionary with label as key and a list of tuple ranges as values
                {
                    "RA": [(start_idx_0, end_idx_0), (start_idx_1, end_idx_1), ... ]
                    "RD": [(), (), ..],
                    "SA": [(), (), ..],
                    ...
                }
    '''
    loaded_labels = [load_leipzig_label_step_indices_from_file(label_f) for label_f in label_files]
    labels_indices_dict = {}
    for label, indices_ranges in loaded_labels:
        if label in ["LW_even", "LWe", "LWu", "LW_uneven"]:
            label = "LW"

        labels_indices_dict[label] = indices_ranges

    return labels_indices_dict

def get_interpolated_df(input_df, num_values = 101):
    '''
    
    '''
    # Uncomment the following lines to get a plot
    # plot_default_gt(input_df)
    # plt.tight_layout()
    # plt.xlabel("Index")
    # plt.show()

    out_df = pd.DataFrame()
    for col in input_df.columns.values:
        out_df[col] = interpolate(input_df[col].values, output_len = num_values)

    # Uncomment the following lines to get a plot
    # plot_default_gt(out_df)
    # plt.tight_layout()
    # plt.xlabel("Gait Cycle [100%]")
    # plt.show()

    return out_df

def find_pattern_in_sequence(state_sequence, pattern):
    '''
    Useful to find typical state machine patterns in a state machine sequence. 

    :param state_sequence:  a list of certain type describing the rule sequence of the data e.g. ['ToBasis', 'ToStanceFun', 'StanceFun', 'Unloaded', 'StanceFun', 'Unloaded', 'StanceFun', 'ToBasis', 'Basis', 'ToStanceFun', 'StanceFun', 'Yielding', 'YieldingEnd', 'ToBasis', ...]
                            look for patterns in this list
                            This list is assumed to be reduced 
    :param state_start_indices: assign the start index to every rule 
    :param pattern: a list of same type as sequence, describing a pattern that should be found. Pattern refers to a small sub-sequence that should be in the state_sequence
                    e.g. ["StanceFlex", "StanceExt", "SwUnlock"]
                    a state in the pattern with the prefix "No_" e.g. "No_StanceFlex" means that every other state except "StanceFlex" is accepted to match the pattern
                    CAUTION: "No_"- state cannot be the first state so far !!! (09.11.2022)
                    CAUTION: "No_"- state can only exists for one state in the pattern so far !!! (09.11.2022)
    :return:    2 lists of tuples with start and end indices of the pattern. The first  the indices given by rules_start_indices
                [(start_idx_1, end_idx_1), (start_idx_2, end_idx_2), ...]
    '''
    pattern_found_indices = []

    pattern = np.array(pattern)
    state_sequence = np.array(state_sequence)

    state_sequence_pattern_start = np.where(state_sequence == pattern[0])[0]

    for start_index_sequence in state_sequence_pattern_start:
        if start_index_sequence + len(pattern) <= len(state_sequence):
            state_sequence_pattern = state_sequence[start_index_sequence:start_index_sequence+len(pattern)]
            
            matches_pattern = True
            
            for a, b in zip(state_sequence_pattern, pattern):
                if b[:3] == "No_":
                    if a == b[3:]:
                        matches_pattern = False
                else: 
                    if a != b:
                        matches_pattern = False

            if(matches_pattern):
                end_idx_sequence = np.amin((start_index_sequence + len(pattern), len(state_sequence)))
                pattern_found_indices.append((start_index_sequence, end_idx_sequence))

        elif pattern[-1][:3] == "No_":
            if start_index_sequence + len(pattern) - 1 <= len(state_sequence):
                state_sequence_pattern = state_sequence[start_index_sequence:start_index_sequence + len(pattern) - 1]

                if (state_sequence_pattern == pattern[:-1]).all():
                    end_idx_sequence = np.amin((start_index_sequence + len(pattern) - 1, len(state_sequence)))
                    pattern_found_indices.append((start_index_sequence, end_idx_sequence))

        else: 
            continue 


    return pattern_found_indices

def compress_state_progression(state_progression):
    '''
    Compress a given (state) progression (list or array). The returned list only contains the state and start index. E.g.
    sequence = ["StanceFlex", "StanceFlex", "StanceFlex", "StanceExt", "StanceExt"] this will return ["StanceFlex", "StanceExt"], [0, 3] 
    
    :param state_progression: list/arr of states. State can be of any type, this function will check if subsequent list entries are similar or not 
    :return:    sequence --> compressed (no subsequent duplicate values )
                sequence_start_indcies --> the starting indices of each sequence (indices referring to state_progression list i.e. the original list)
    '''
    state_progression = np.array([-1] + list(state_progression))
    sequence_start_indices = np.where(state_progression[1:] != state_progression[:-1])[0]
    sequence_start_indices += 1

    sequence = [state_progression[idx] for idx in sequence_start_indices]

    sequence_start_indices -= 1
    # if sequence_start_indices[0] == 1:
    #     sequence_start_indices[0] = 0

    return sequence, sequence_start_indices

def get_activity_from_int_code_chris(activity_code: int):
    '''
    return the "string-activity" based on the int activity code. See ACTIVITY_TYPE_CODE_DICT_CHRIS
    '''
    return list(ACTIVITY_TYPE_CODE_DICT_CHRIS.keys())[np.where(np.array(list(ACTIVITY_TYPE_CODE_DICT_CHRIS.values())) == activity_code)[0][0]]


def get_time_in_sec_from_df(df): 
    '''
    df hast to has the column CYCLETIME. This is an indicator for the time between two samples (in milliseconds), i.e. the sampling frequency can also be derived.
    The time in seconds can be achieved via integration of this column. 

    :param df: input df with CYCLETIME_COL column. C
    :return: array with time indicated as seconds
    '''

    time_in_sec = df['CYCLETIME'].cumsum() / 1000

    return time_in_sec

def get_time_in_ms_from_df(df):
    '''
    See get_time_in_sec_from_df. Returns time indicated in ms
    '''
    time_in_sec = df['CYCLETIME'].cumsum() / 1000

    return time_in_sec    

def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format --> useful for the print platform specifications method
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

def print_platform_specifications(output_specs_file):
    '''
    Print information of the used machine to a txt-file.

    :param output_specs_file: txt file to which the information should be printed
    '''
    original_stdout = sys.stdout

    pardir = Path(output_specs_file).parent
    if not os.path.exists(pardir):
        os.makedirs(pardir)

    with open(output_specs_file, 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.
    
        print("="*40, "System Information", "="*40)
        uname = platform.uname()
        print(f"System: {uname.system}")
        print(f"Node Name: {uname.node}")
        print(f"Release: {uname.release}")
        print(f"Version: {uname.version}")
        print(f"Machine: {uname.machine}")
        print(f"Processor: {uname.processor}")

        # Boot Time
        print("="*40, "Boot Time", "="*40)
        boot_time_timestamp = psutil.boot_time()
        bt = datetime.fromtimestamp(boot_time_timestamp)
        print(f"Boot Time: {bt.year}/{bt.month}/{bt.day} {bt.hour}:{bt.minute}:{bt.second}")

        # let's print CPU information
        print("="*40, "CPU Info", "="*40)
        # number of cores
        print("Physical cores:", psutil.cpu_count(logical=False))
        print("Total cores:", psutil.cpu_count(logical=True))
        # CPU frequencies
        cpufreq = psutil.cpu_freq()
        print(f"Max Frequency: {cpufreq.max:.2f}Mhz")
        print(f"Min Frequency: {cpufreq.min:.2f}Mhz")
        print(f"Current Frequency: {cpufreq.current:.2f}Mhz")
        # CPU usage
        print("CPU Usage Per Core:")
        for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
            print(f"Core {i}: {percentage}%")
        print(f"Total CPU Usage: {psutil.cpu_percent()}%")

        # Memory Information
        print("="*40, "Memory Information", "="*40)
        # get the memory details
        svmem = psutil.virtual_memory()
        print(f"Total: {get_size(svmem.total)}")
        print(f"Available: {get_size(svmem.available)}")
        print(f"Used: {get_size(svmem.used)}")
        print(f"Percentage: {svmem.percent}%")
        print("="*20, "SWAP", "="*20)
        # get the swap memory details (if exists)
        swap = psutil.swap_memory()
        print(f"Total: {get_size(swap.total)}")
        print(f"Free: {get_size(swap.free)}")
        print(f"Used: {get_size(swap.used)}")
        print(f"Percentage: {swap.percent}%")

        print("\n\n\n\n\n")

    sys.stdout = original_stdout


def print_used_features(output_file, features_list):
    '''
    Print information of the used features to the text file

    :param output_file: txt file to which the information should be printed
    :param features_list: list of feature names as str
    '''
    features_list = list(features_list)
    features_list.sort()

    original_stdout = sys.stdout
    mode = "a" if os.path.exists(output_file) else "w"
    with open(output_file, mode) as f:
        sys.stdout = f # Change the standard output to the file we created.
    
        print(f"Used Features ({len(features_list)}):")
        for feature in features_list:
            print(f"{feature}")
        
        print("\n\n")

    sys.stdout = original_stdout

def print_to_file(output_file, text: str):
    '''
    Print some text to some output-txt file
    '''
    mode = "a" if os.path.exists(output_file) else "w"

    original_stdout = sys.stdout
    with open(output_file, mode) as f:
        sys.stdout = f # Change the standard output to the file we created.
        print(text)
        sys.stdout = original_stdout


def close_activity_gaps(activity_stream, max_gap = 5, max_gap_sitting = 30):
    '''
    SMOOTHES THE ACTIVITY STREAM 

    Close activity gaps, i.e. if there is a sequency of different activity shorter than max_gap samples between the same activity, close this gap 
        somewhat similar to a closing operation in images
        E.g.    - max_gap = 5 
                - input activity_stream     = [100, 100, 100,   0,   0,   0, 100, 100]
                - output activity_stream    = [100, 100, 100, 100, 100, 100, 100, 100]
    
    :param activity_stream: array representing an activity stream e.g. [0, 0, 0, 0, 100, 100, 100, 100, ....400, 400, 400, 200, 200, 200, 200]
    :param max_gap: maximumg gap to close for all activities
    :param max_gap_sitting: maximum gap to close if "surrounding" activity is sitting (now 30 corresponding to 0.3 seconds)
    '''

    activity_progression = activity_stream[1:] - activity_stream[:-1]

    indices = np.where(activity_progression != 0)[0]

    # activity_stream_before = activity_stream.copy()

    for activity_start_idx, activity_end_idx in zip(indices[:-1], indices[1:]):

        if activity_end_idx - activity_start_idx <= 1:
            # Close gap if only one different sample is in between
            activity_stream[activity_start_idx : activity_end_idx + 1] = activity_stream[activity_start_idx - 1]

        elif activity_stream[activity_start_idx - 1] == activity_stream[activity_end_idx + 1]:
            # Apply a larger gap if sitting activity is "performed"
            gap = max_gap_sitting if activity_stream[activity_start_idx - 1] == ACTIVITY_TYPE_CODE_DICT_CHRIS["sitting"] else max_gap

            if activity_end_idx - activity_start_idx < gap:
                activity_stream[activity_start_idx : activity_end_idx + 1] = activity_stream[activity_start_idx - 1]
        else: 
            continue

    return activity_stream 


def merge_dataframes(df_1, df_2):
    '''
    Merge two dataframes and drop duplicate columns.

    :return: the merged dataframe
    '''
    df_1 = df_1.merge(df_2, left_index = True, right_index = True, suffixes=("", "_delete"))

    cols_delete = [c for c in df_1.columns if "_delete" in c]
    cols_keep = [c.replace("_delete", "") for c in cols_delete]
    # assert(np.all(["delete" not in c for c in df.columns.values]))

    assert(np.all( df_1[c_del].equals(df_1[c_keep]) for c_del, c_keep in zip(cols_delete, cols_keep) ))
    for c in cols_delete:
        df_1.drop(c, inplace = True, axis = 1)

    return df_1


def label_dict_from_labels_list(labels: list):
    '''
    :param labels: list of labels, label is an integer value
    '''
    label_dict_f = {}
    total_count_patient = 0

    for key in list(ACTIVITY_CODE_TYPE_DICT_CHRIS.keys()):
        num = labels.count(key)
        label_dict_f[key] = num

        total_count_patient += num 
    
    return label_dict_f


### SOME USEFUL METHODS FOR HANDLING FILES AND DIRECTORIES
def remove_empty_dir(directory):
    '''
    If the path exists, is a directory and is empty --> remove
    '''
    if os.path.exists(directory):
        if os.path.isdir(directory):
            if len(os.listdir(directory)):
                os.rmdir(directory)
    
def create_dir(directory):
    '''
    If the directory does not exist --> create
    '''
    if not os.path.exists(directory): 
        os.makedirs(directory)

def move_files(input_directory, output_directory):
    '''
    Move all files from the input directory to the output directory
    '''
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # fetch all files
    for file_name in os.listdir(input_directory):
        # construct full file path
        source = input_directory + file_name
        destination = output_directory + file_name
        # move only files
        shutil.move(source, destination)

def clean_up_directory(directory):
    '''
    Delete all files in a given directory.
    '''
    for f in os.listdir(directory):
        os.remove(os.path.join(directory, f))


def save_activity_code_type_dict(activity_code_type_dict, output_dir, out_basename = "activity_code_type_dict"):
    '''
    Save the activity code type dictionary in the output directory provided as pkl object
    and additionally prints the values to a text file

    :param activity_code_type_dict: dictionary, integer code as key and string label as item
    :param output_dir: where to save

    '''
    pkl_output_f = os.path.join(output_dir, f"{out_basename}.pkl")
    txt_output_f = os.path.join(output_dir, f"{out_basename}.txt")

    # if output exists --> check if the same mapping/dict has been used
    if os.path.exists(pkl_output_f):
        existing_activity_code_type_dict = get_dir_activity_code_type_dict(output_dir, basename = out_basename, ignore_discard_class=False)
        if existing_activity_code_type_dict != activity_code_type_dict: 
            print("Trying to save dataset with a different mapping than existing one. Chose a different output directory or delete the existing activity type code dictionary (and data associated with this) in your chosen output directory.")
            sys.exit()

    print_activity_code_type_dict_to_file(activity_code_type_dict, txt_output_f)

    with open(pkl_output_f, 'wb') as f:
        pkl.dump(activity_code_type_dict, f)

def print_activity_code_type_dict_to_file(activity_code_type_dict, txt_output_f):
    '''
    '''

    original_stdout = sys.stdout
    
    if os.path.exists(txt_output_f):
        return

    with open(txt_output_f , "w") as f:
        sys.stdout = f # Change the standard output to the file we created.

        for key, item in activity_code_type_dict.items():
            print(f"{key}\t{item}")

    sys.stdout = original_stdout
    return 

def get_dir_activity_code_type_dict(output_dir, basename = "activity_code_type_dict", ignore_discard_class = True):
    ''' 
    Load the activity code type dict used for generating the data in this directory
    '''
    pkl_output_f = os.path.join(output_dir, f"{basename}.pkl")

    with open(pkl_output_f, 'rb') as f:
        activity_code_type_dict = pkl.load(f)

    if ignore_discard_class:
        activity_code_type_dict = {key:val for key, val in activity_code_type_dict.items() if val != DISCARD_CLASS_TYPE}    
        
    return activity_code_type_dict


def swap_key_entry_dict(input_dict):
    '''
    For an input dictionary convert the key and item value.

    E.g. {
        key1: value1,
        key2: value2
        }
    
    will be returned as

    {   
        value1: key1,
        value2: key2
    }
    '''
    output_dict = {}

    for k, v in input_dict.items():
        output_dict[v] = k
    
    return output_dict


def apply_label_mapping(predictions, label_mapping):
    '''
    :param label_mapping: dictionary with input label as key and output label as value
    '''
    # Map the predictions
    predictions = [label_mapping[p] for p in predictions]

    return predictions

def label_mapping_dict_to_df(label_mapping: dict):
    '''
    Convert the label mapping dict to dataframe
    '''
    data = np.array((list(label_mapping.keys()), list(label_mapping.values())))
    data = np.transpose(data)
    label_mapping_df = pd.DataFrame(data = data, columns = ["label_in", "label_out"], index=None)

    return label_mapping_df

def apply_mapping_to_dict_keys(initial_dict, key_mapping_dict):
    '''
    :param initial_dicitonary:  dictionary for which the key values should be changed
                                e.g. a dictionary with class occurences where the activity label code is the key  
    :param key_mapping:         dictionary where the key is the current key in initial_dict and the value is the desired, mapped key 
                                e.g. the activity_code_type_dict

    :return: output dictioanry with new key values
    '''
    output_dict = {}

    for k, val in initial_dict.items(): 

        output_dict[key_mapping_dict[k]] = val 

    return output_dict

def scale_dictionary(input_dict, min_val, max_val):
    '''
    Scale the values of a dicitonary.
    '''
    # set the minimum transition probability
    scaler = MinMaxScaler(feature_range=(min_val, max_val))
    values_arr = np.array(np.array(list(input_dict.values()))).reshape(-1, 1)
    scaler.fit(values_arr)
    values_arr = scaler.transform(values_arr)

    output_dict = dict(zip(input_dict.keys(), values_arr))

    return output_dict

def df_to_dict(df, key_col:str, value_col:str):
    '''
    From a dataframe extract a dictionary. key_col is the column values used as the key in dictionary
    '''
    assert(key_col in df.columns.values)
    assert(value_col in df.columns.values)

    dict_out = {}
    for idx, row in df.iterrows():
        dict_out[row[key_col]] = row[value_col] 

    return dict_out



