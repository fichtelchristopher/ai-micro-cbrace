''' 
Master thesis project of Christopher Fichtel (christopher.fichtel@ottobock.com) started in Oct 2022
AI in operating ortheses

File created 10 Oct 2022

File for general file handling operations. 
'''
import glob
import os
from pathlib import Path
import numpy as np 


def get_set_rule_output_fname(output_dir, f):
    '''
    Determine output file
    
    :param rule_id_output_dir: where to save the data
    :param f: input filename (i.e. the file where input data is loaded from)
    :return: output_filename
    '''
    set_rule_out_fname = os.path.join(output_dir, os.path.basename(f))
    set_rule_out_fname = set_rule_out_fname.replace(".txt", "_.txt")
    set_rule_out_fname = set_rule_out_fname.replace(".csv", "_.txt")
    set_rule_out_fname = set_rule_out_fname.replace(".mat", "_.txt")
    
    return set_rule_out_fname

def get_activity_progression_output_fname(output_dir, f, starting_time = None, range = None, additional_str = None):
    '''
    Return the filename for a numpy-file where the activities-array will be saved. 
    
    :param output_dir: where to save the data
    :param f: input filename (i.e. the file where input data is loaded from) --> will use file's basename
    :param range:   either None i.e. whole data file is loaded 
                    or a tuple indicating the start and end idx
    :return: filename in the "*.npy" format
    '''    
    activities_out_fname = get_subject_output_name(output_dir=output_dir, f = f, output_ending="npy", range = range, additional_str = None)

    return activities_out_fname

def get_subject_output_name(output_dir, f, output_ending = "csv", starting_time = None, range = None, additional_str = None):
    '''
    Return output filenames for a subject. Modifies the input filename (might change ending, adds starting time, range or additional str information)
    
    :param output_dir: where to save the data
    :param f: input filename (i.e. the file where input data is loaded from) --> will use file's basename
    :param starting_time: if supplied --> add to filename
    :param range:   either None i.e. whole data file is loaded 
                    or a tuple indicating the start and end idx
    :param additional_str: if supplied --> add to filename
    
    :return: filename in the "*.output_ending" format (used so far "npy" and "csv")
            format: {basename}_{starting_time}_{range}_{additional_str}.{output_ending}
    '''
    activities_out_fname = os.path.join(output_dir, os.path.basename(f))
    file_ending = activities_out_fname.split(".")[-1]
    activities_out_fname = activities_out_fname.replace(f".{file_ending}", f".{output_ending}")

    if starting_time: 
        activities_out_fname = activities_out_fname.replace(f".{output_ending}", f"_{starting_time}.{output_ending}")
    if range: 
        activities_out_fname = activities_out_fname.replace(f".{output_ending}", f"_{range[0]}-{range[1]}.{output_ending}")
    if additional_str: 
        activities_out_fname = activities_out_fname.replace(f".{output_ending}", f"_{additional_str}.{output_ending}")
    
    return activities_out_fname

def get_subject_name(f):
    '''
    From a file to an subject file extract the subject name.
    '''
    subject_name = os.path.basename(f)

    for sep in ["."]:
        subject_name = subject_name.split(sep)[0]

    return subject_name

def get_subject_df_out_fname(output_dir, f, range = None):
    '''
    Return the filename for a csv-file where a subjects preprocessed and ground truth df is saved. 
    
    :param output_dir: where to save the data
    :param f: input filename (i.e. the file where input data is loaded from) --> will use file's basename
    :param range:   either None i.e. whole data file is loaded 
                    or a tuple indicating the start and end idx
    :return: filename in the "*.csv" format
    '''    

    subject_df_out_fname = get_subject_output_name(output_dir=output_dir, f = f, output_ending="pkl", range = range, additional_str = "gt")

    return subject_df_out_fname

def get_subject_df_out_fname_leipzig(output_dir, f, subject_name = ""):
    '''
    Return the filename for a csv-file where a subjects preprocessed and ground truth df is saved. 
    
    :param output_dir: where to save the data
    :param f: input filename of the leipzig data, a file is saved under .../.../{subject_name}/labeled/{time_stamp}.txt
                thus the subject directory is the parent of the parent directory 

    :return: filename in the "*.csv" format
    '''    
    if len(subject_name) == 0:
        subject_name = os.path.basename(Path(f).parent.parent) 
    # f_basename = os.path.basename(f).replace(".txt", ".csv")

    get_subject_df_out_fname = os.path.join(output_dir, f"{subject_name}.pkl")#_{f_basename}")

    return get_subject_df_out_fname

def get_subject_rule_progression_output_fname(output_dir, f, range):
    '''
    Return the filename for a npy file where a subjects rule progression is saved. 
    
    :param output_dir: where to save the data
    :param f: input filename (i.e. the file where input data is loaded from) --> will use file's basename
    :param range:   either None i.e. whole data file is loaded 
                    or a tuple indicating the start and end idx
    :return: filename in the "*.npy" format
    '''    
    rule_progression_out_fname = get_subject_output_name(output_dir=output_dir, f = f, output_ending="npy", range = range, additional_str = "rule_progression")
    return rule_progression_out_fname

def get_subject_activity_statistic_output_fname(output_dir, f, range = None):
    '''
    Return the filename for a text-file where the subject activity statistics will be saved. 
    
    :param output_dir: where to save the data
    :param f: input filename (i.e. the file where input data is loaded from) --> will use file's basename
    :param range:   either None i.e. whole data file is loaded 
                    or a tuple indicating the start and end idx (start_idx, end_idx)
    :return: filename in the "*.npy" format
    ''' 
    subject_activities_statistics_out_fname = get_subject_output_name(output_dir=output_dir, f = f, output_ending="txt", range = range, additional_str = "subject_activity_statistics")

    return subject_activities_statistics_out_fname

def get_window_size_str(w_size):
    '''
    :param w_size: window size in samples
    :return: a string for e.g. saving in directory
    '''
    return f"w{w_size}"

def get_step_size_str(s_size):
    '''
    :param s_size: step size in samples
    :return: a string for e.g. saving in directory
    '''
    return f"s{s_size}"

def get_num_epochs_str(epochs):
    '''
    '''
    return f"e{epochs}"

def get_feature_size_str(num_features):
    '''
    :param n_features: number of features used
    :return: a string for e.g. saving in directory
    '''
    return f"nfeat{num_features}"

def get_train_split_str(train_split):
    '''
    :param train_split: range in the range of (0, 1) indicating the training split
    :return: a string for e.g. saving in directory
    '''
    return f"TRAIN_{train_split}"

def get_val_split_str(val_split):
    '''
    :param train_split: range in the range of (0, 1) indicating the training split
    :return: a string for e.g. saving in directory
    '''
    return f"VAL_{val_split}"


def get_train_dir(train_val_dir, train_split):
    '''
    
    '''
    train_split = np.round(train_split, 2)
    train_dir = os.path.join(train_val_dir, get_train_split_str(train_split))
    return train_dir

def get_val_dir(train_val_dir, val_split):
    '''
    
    '''
    val_split = np.round(val_split, 2)
    val_dir = os.path.join(train_val_dir, get_val_split_str(val_split))
    return val_dir

def get_models_predictions_evaluation_out_dir(base_out_dir, model_type, window_step_size_samples, num_selected_feature_cols, window_size_samples = 10, fixed_window = False):
    '''
    
    '''
    if fixed_window:
        return os.path.join(base_out_dir, f"{model_type}/{get_window_size_str(window_size_samples)}_{get_step_size_str(window_step_size_samples)}_{get_feature_size_str(num_selected_feature_cols)}") 
    else:
        return os.path.join(base_out_dir, f"{model_type}/{get_step_size_str(window_step_size_samples)}_{get_feature_size_str(num_selected_feature_cols)}") 

def get_model_out_name(model_output_dir, file, train_per_file = True):
    '''
    :param split_mode: either "loso" or "standard"
                        "loso" means that the file given in the file parameter of this method is considered the test file and 
                            the model has been trained on the other files of the directory
                        "standard" means that the model has been trained exactly and only on this file
    '''
    model_basename = os.path.basename(file).replace("_gt_reduced.csv", f"_gt.csv")
    model_basename = model_basename.replace("_gt_features_normalized.csv", "_gt.csv")
    if train_per_file:
        model_basename = (model_basename).replace("_gt.csv", "_trained_on.pkl")
    else:
        model_basename = (model_basename).replace("_gt.csv", "_left_out.pkl")
    
    model_out_fname = os.path.join(model_output_dir, model_basename)

    return model_out_fname 

def get_nn_model_out_name(model_output_dir, file, train_per_file = False):
    '''
    '''
    model_out_fname = get_model_out_name(model_output_dir, file, train_per_file)

    model_out_fname = model_out_fname.replace(".pkl", ".h5")
    
    return model_out_fname

def get_nn_models_predictions_evaluation_out_dir(base_out_dir, model_type, window_size_samples, window_step_size, epochs):
    '''
    
    '''
    return os.path.join(base_out_dir, f"{model_type}/{get_window_size_str(window_size_samples)}_{get_step_size_str(window_step_size)}_{get_num_epochs_str(epochs)}") 

def get_nn_predictions_out_name(predictions_output_dir, file, train_per_file = False):
    '''
    :param val_file: the filename that has been left for for LOSO training  
    '''
    predictions_out_fname = get_model_out_name(predictions_output_dir, file, train_per_file)

    predictions_out_fname = predictions_out_fname.replace(".pkl", "_predictions.csv")
    
    return predictions_out_fname

def get_model_basename_from_leave_out_test_file(test_file):
    
    basename = os.path.basename(test_file).replace("_gt_reduced.pkl", "_gt.pkl")
    basename = basename.replace("_gt_features_normalized.pkl", "_gt.pkl")
    basename = basename.replace("_gt.pkl", "") + "_leave_out"
    basename = basename.replace(".pkl", "")

    return basename

def get_model_name_from_leave_out_test_file(model_type, test_file):

    return f"{model_type}_{get_model_basename_from_leave_out_test_file(test_file)}"


def get_leipig_files_from_dir(input_dir):
    '''
    Return all the files that match the leipzig pattern. 
    '''
    return glob.glob(input_dir + "/*[0-9].pkl")


def get_aos_files_from_dir(input_dir):
    '''
    Return all the files that match the aos pattern
    '''
    return glob.glob(input_dir + "/AOS*.pkl")

def get_dataset_dirname_str(dataset_num: int):
    '''
    '''
    return f"DS{dataset_num}"