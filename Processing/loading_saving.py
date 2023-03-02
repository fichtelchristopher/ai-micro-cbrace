''' 
Master thesis project of Christopher Fichtel started in Oct 2022
AI in operating ortheses

File created 06 Oct 2022

Code Base for loading matlab and csv data files
'''

from cProfile import label
import os 
import glob 
import scipy.io
import pandas as pd
import numpy as np
import pickle
import sys
import csv
from Configs.namespaces import *
from Configs.shallow_pipeline_config import SHALLOW_IMPLEMENTATIONS
from Configs.nn_pipeline_config import NN_IMPLEMENTATIONS
from Configs.pipeline_config import *
from Processing.file_handling import get_leipig_files_from_dir, get_aos_files_from_dir

def load_data_file(filepath, range = None): 
    '''
    Depending on the file ending, load a participants data file. 
    Supported formats: "csv", "mat" and "txt"

    :param filepath:
    :param range: optional, default is None --> load the whole file 
            if required: supply range as tuple of (start_idx, end_idx)
    :return: dataframe of the loaded data
    '''
    if filepath.endswith(".csv"):
        df = load_csv_file(filepath, range = range)
    elif filepath.endswith(".mat"):
        df = load_mat_file(filepath, range = range)
    elif filepath.endswith(".txt"):
        df = load_txt_file(filepath)
    else:
        print("Unsupported file-ending. Supported endings are 'csv', 'mat' and 'txt'.")

    return df

def load_mat_file(filepath, drop_nan_cols = False, range = None):
    '''
    Load a matlab file and return it as a pandas dataframe.

    :param filepath: str, path to the matlab-file
    :param drop_nan_cols:   if columns containing nan values should be deleted
                            e.g. time in matlab files sometimes is bad represented and contains nan in its column
    :param range: optional, default is None --> load the whole file 
            if required: supply range as tuple of (start_idx, end_idx)
    
    :return: pandas dataframe
    '''
    assert (os.path.isfile(filepath) and str(filepath)[-4:] == ".mat")

    mat_f = scipy.io.loadmat(filepath, squeeze_me = True)

    header      = mat_f["AOSheader"]  # shape 1 x cols
    data_array  = mat_f["AOSdata"]    # shape (data length) x cols
    if range: 
        data_array = data_array[range[0]:range[1], :]
    file        = mat_f["AOSfiles"]   # gives the corresponding csv base file
    
    # Delete columns containing nan values (if desired)
    if drop_nan_cols:
            is_nan_col = np.isnan(data_array).any(axis=0)
            nan_col_indices = np.where(is_nan_col == True)
            header = np.delete(header, nan_col_indices)
            data_array = np.delete(data_array, nan_col_indices, axis = 1)

    # Delet invalid columns where the column name is no string --> invalid and not possible to create a dataframe with such column 'names'
    is_nostring_col    = np.array([type(col) != str for col in header])
    is_nostring_col    = np.where(is_nostring_col == True)
    header = np.delete(header, is_nostring_col)

    data_array = np.delete(data_array, is_nostring_col, axis = 1)
    df = pd.DataFrame(data = data_array, columns = header, dtype = float)

    return df


def load_csv_file(filepath, contains_cols = [], sep = ";", range = None): #"DDD_VEL_X", "DDD_VEL_Y", "DDD_VEL_Z"]):
    '''
    Load a csv file and return it as a pandas dataframe.

    :param filepath: str, path to the csv-file
    :param contains_cols: 
                    a list of column names that the csv has to contain --> this does not mean 
                    that only these columns will be loaded, but used for determining the proper
                    loading of the header row
                    often the header row is not in the first row of a csv file
    :param range: optional, default is None --> load the whole file 
            if required: supply range as tuple of (start_idx, end_idx)

    :return: pandas dataframe
    '''
    assert (os.path.isfile(filepath) and str(filepath)[-4:] == ".csv")

    # idx is the value where the header line starts
    # i.e. the first row in the csv that contains a seperator
    with open(filepath, "r") as fin:
        reader = csv.reader(fin)
        idx = next(idx for idx, row in enumerate(reader) if len(row) > 1)
    skiprows = idx - 1

    if range:
        df = pd.read_csv(filepath, skiprows = skiprows, nrows=range[1] - range[0], sep = sep)
    else:
        df = pd.read_csv(filepath, skiprows = skiprows, sep = sep)

    assert set(contains_cols).issubset(set(df.columns.values))

    return df

def load_txt_file(filepath, sep = "\t"):
    '''
    Load a txt file and return it as a pandas dataframe.

    Currently implemented for the structure of the text files available in Leipzig2018Messungen directory
    e.g. Ottobock SE & Co. KGaA\Deep Learning - General\Knowledgebase\Leipzig2018Messungen\DM0104\labeled\2018-05-16_15-43-05-142 

    :param filepath: str, path to the txt-file
    :param sep: separator used in the text file, by default tab encoded as "\t"
    :return: pandas dataframe
    '''


    col_names = []
    skiprows = 0
    endrow_data = 0
    
    with open(filepath) as f:
        next_header = False

        for num, line in enumerate(f):
            if next_header: 
                col_names = extract_cols_from_excel_header(line, sep = sep)
                next_header = False

            if line.startswith("[ExcelHeader]"):
                # next line contains header data
                next_header = True

            if line.startswith("[MeasurementData]"):
                skiprows = num
            
            if line.startswith("[/MeasurementData]"):
                endrow_data = num # the first row after measurement that contains no data 
                break

    assert(skiprows > 0)    # otherwise check if "[MeasurementData]" exists in the txt file --> this will be used as indicator for 
    assert(endrow_data > 0) # otherwise check if "[/MeasurementData]" exists in the txt file --> this will be used as indicator for 
    assert(len(col_names) > 0)

    df = pd.read_table(filepath, header = 0, names = col_names, sep = "\t", skiprows = skiprows, nrows = endrow_data - skiprows - 1)

    df = df.replace({',': '.'}, regex=True)

    for s in RAW_SIGNAL_COLS:
        df[s] = df[s].astype("float")

    return df


def extract_cols_from_excel_header(line, sep = "\t"):
    '''
    Header in txt file is seperated with tab
    '''
    data_cols = line.split(sep)
    return data_cols 


def get_data_files_leipzig_dict(data_dir):
    '''
    Data is under "Ottobock SE & Co. KGaA\Deep Learning - General\Knowledgebase\Leipzig2018Messungen"
    Per patient (7) return the text files, patients are imported from namespaces.py

    Each txt-matfile can be loaded as a pd dataframe using the load_txt_file method.
    
    :param data_dir: path to the Leipzig2018Messungen directory
    :param return: dictionary 
            {
                "patientID1": [List, of, paths, to, textfiles], 
                "patientID2": [ ... ],
                "patientID3": [ ... ], 
                ...
            }

    :note: if not a dict is required but a list --> list(data_files_dict.values())       
    '''
    data_files_dict = {}

    for patient in LEIPZIG_PARTICIPANTS:
        labeled_dir = os.path.normpath(os.path.join(data_dir, f"{patient}/labeled"))
        txt_files = glob.glob(labeled_dir + "/*.txt")

        data_files_dict[patient] = [os.path.join(labeled_dir, t_f) for t_f in txt_files]

    return data_files_dict

def get_leipzig_label_files(f, data_dir = None):
    '''
    Return the label files associated with a data file. These are files under the "labeled" data directory in Leipzig2018Messungen.

    :param f: file for which the label files should be returned
    :param data_dir: directory where to look for the files
                    None --> take directory of data dir as default
    
    :return: list of data files (based on this list one can extract labels)
    '''
    assert(os.path.exists(f))

    if data_dir is None: 
        data_dir = os.path.dirname(f)

    basename = get_leipzig_basename_from_filepath(f)
    label_files = glob.glob(data_dir + f"/{basename}*.mat")

    # Filte copy of files via checking if "Kopie" is in the filename
    label_files = [f for f in label_files if "Kopie" not in f]

    return label_files

def get_leipzig_basename_from_filepath(f):
    '''
    Use this method to find the basename pattern in order to determine corresponding mat files containing label's indices.

    :param f:   filepath e.g. "path/to/filedir/2018-05-16_15-43-05-142.txt" 
    :basename:  for the given example this will return "2018-05-16_15-43-05"
                The label's filename is in the format of "2018-05-16_15-43-05-label_XX.mat" i.e. basename + "-label_XX.mat"
    '''
    basename = os.path.basename(f)
    split_idx = basename.rindex("-")
    basename = basename[:split_idx]

    return basename

def load_leipzig_label_step_indices_from_file(f):
    '''
    :param f: path to the label file (.mat) e.g. /path/to/2018-05-16_15-43-05-label_SD.mat
    :return: label and list of indices for the step
    '''
    mat_f = scipy.io.loadmat(f, squeeze_me = True)

    indices = mat_f["step"]
    if len(indices.shape) == 1:
        indices = np.expand_dims(indices, 0)

    f = f.split(".")[-2]
    label = f.split("label_")[-1]

    indices = [(idx[0], idx[1]) for idx in indices]

    return label, indices

def load_leipzig_gt_from_label_file(f):
    '''
    :param f: path to the label file (.mat) e.g. /path/to/2018-05-16_15-43-05_gtLabels.mat
    :return: array with ground truth as integer
    
    '''
    mat_f = scipy.io.loadmat(f, squeeze_me = True)
    
    gt = np.array(mat_f["gt"])
    
    return gt


def get_configs_from_output_dir(model_output_dir):
    '''
    The output directory when running the ai pipeline of training, prediction and evaluation looks like dir

    ___output_dir_path
        |
        |___ Training_{STEP_SIZE}           --> directory to save training results
        |
        |___ Prediction_{STEP_SIZE_TEST}    --> directory to save prediction results
        |
        |___ Evaluation_{STEP_SIZE_TEST}    --> directory to save evaluation results
        |
        |___ pipeline_config.pkl            --> pipeline configuaration that can be loaded
        |
        |___ classifier_config.pkl          --> classifier configuraiton that can be loaded


    :param return:  pipeline_config, classifier_config
                    returns the pipeline and classifier configuration loaded dictionaries
                    if the path to the config does not exists --> return None for this config
    '''
    assert(os.path.exists(model_output_dir))

    pipeline_config_file = os.path.join(model_output_dir, "pipeline_config.pkl")
    if os.path.exists(pipeline_config_file):
        with open(pipeline_config_file, "rb") as f:
            pipeline_config = pickle.load(f)
    else:
        pipeline_config = None

    classifier_config_file = os.path.join(model_output_dir, "classifier_config.pkl")
    if os.path.exists(classifier_config_file):
        with open(classifier_config_file, "rb") as f:
            classifier_config = pickle.load(f)
    else:
        classifier_config = None

    return pipeline_config, classifier_config

def get_train_test_files(classifier_type, activities_output_dir = ACTIVITIES_OUTPUT_DIR, features_normalized_dir = FEATURES_NORMALIZED_DIR, database = DATABASE):
    '''
    Return a list of files that can be used for training / testing.

    if the classifier of type NN:                   --> return the files with the raw sensor data
    if the classifier of type ShallowClassifier:    --> return the files with the normalized feature vectors
    
    '''

    if classifier_type.upper() in NN_IMPLEMENTATIONS:
        activities_dir = activities_output_dir
    elif classifier_type.lower() in SHALLOW_IMPLEMENTATIONS:
        activities_dir = features_normalized_dir
    else:
        print(f"Invalid {classifier_type}")

    if database == "aos":
        files = glob.glob(activities_dir + "/AOS*.pkl")
    else:
        files = glob.glob(activities_dir + "/*[0-9].pkl" )

    return files

def save_as_pickle(x, fname):
    '''
    :param x: the (python) object to be saved
    :param fname: the filename, has to end with .pkl
    '''
    assert(fname.endswith(".pkl"))

    with open(fname, 'wb') as f:
        pickle.dump(x, f)


def load_from_pickle(fname):
    '''
    From a pickle filepath load the object and return it.
    :param fname: the filename, has to end with .pkl
    '''
    assert(fname.endswith(".pkl"))

    
    try:
        with open(fname, 'rb') as f:
            x = pickle.load(f)
    except Exception as e: 
        print(f"Cannot load {fname}")
        print(e)
        sys.exit()

    return x