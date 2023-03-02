''' 
Master thesis project of Christopher Fichtel started in Oct 2022
AI for microprocessor-controlled knee joints

File created 09 Nov 2022

Code Base for generation the train validation and test dataset
'''
import numpy as np
import pandas as pd

import glob
import os
import sys

from Misc.utils import *
from Processing.loading_saving import *
from Processing.file_handling import * 
from AOS_SD_analysis.AOS_Rule_Set_Parser import * 
from Configs.pipeline_config import *
from Configs.shallow_pipeline_config import *

from Processing.loading_saving import *

from Misc.dataset_utils import *
from Misc.ai_utils import *

from Misc.feature_extraction import *
from Analysis.aos_data_gt_activity_analysis import * 

from Processing.file_handling import get_leipig_files_from_dir, get_aos_files_from_dir


def generate_features(input_dir, output_dir, signals, window_size_samples, metrics, features_dir = None, database = DATABASE):

    '''
    For all input files in input directory calculate the feature files

    :param features_dir: the directory containing the "big" feature file --> check if here the file already exists and if features are missing
    ''' 
    if DATABASE == "aos":
        files = get_leipig_files_from_dir(aos_dir)
    else:
        files = get_leipig_files_from_dir(input_dir)

    for fname in files: 
        activities_features_out_fname_per_w = os.path.join(output_dir,os.path.basename(fname).replace("_gt_reduced", "_gt"))
        activities_features_out_fname_per_w = activities_features_out_fname_per_w.replace("_gt", "_gt_features") # in case of reduced dataframe
        
        required_feature_cols_out = get_output_feature_cols(signals, [window_size_samples], metrics = metrics)

        if type(features_dir) != type(None):
            activities_features_out_fname = os.path.join(features_dir, os.path.basename(activities_features_out_fname_per_w))
            if os.path.exists(activities_features_out_fname):
                if activities_features_out_fname.endswith(".csv"):
                    features_df = pd.read_csv(activities_features_out_fname, index_col = INDEX_COL, nrows = 1)
                else:
                    features_df = load_from_pickle(activities_features_out_fname).iloc[:1]
                if set(required_feature_cols_out).issubset(set(list(features_df.columns.values))):
                    continue

        # Check if the feature file for this window size has been created and contains all the required feature columns
        # If not --> add the missing columns
        if os.path.exists(activities_features_out_fname_per_w):
            features_df_per_w = load_from_pickle(activities_features_out_fname_per_w).iloc[:1]
            if set(required_feature_cols_out).issubset(set(list(features_df_per_w.columns.values))):
                continue

        print(f"Calculate features for input file {fname} and window size {window_size_samples}")
        
        if fname.endswith(".csv"):
            data_df = pd.read_csv(fname, index_col = INDEX_COL)
        else:
            data_df = load_from_pickle(fname)
            data_df.set_index(INDEX_COL)

        if os.path.exists(activities_features_out_fname_per_w):
            # features_df_per_w = pd.read_csv(activities_features_out_fname_per_w, index_col = INDEX_COL)
            features_df_per_w = load_from_pickle(activities_features_out_fname_per_w)
            features_df_per_w.set_index(INDEX_COL)
        else:
            features_df_per_w = data_df[[LABEL_COL, ACTIVITY_IDX_COL]]
            assert(set(list(signals)).issubset(set(list(data_df.columns.values))))

        existing_feature_cols = features_df_per_w.columns.values

        for act_index in list(set(list(data_df[ACTIVITY_IDX_COL].values))):
            data_df_indexed = data_df.loc[data_df[ACTIVITY_IDX_COL] == act_index]

            data_df_indexed = data_df_indexed[signals]

            # Generate features, select only those columns for future
            data_df_indexed, generated_feature_cols = feature_generation(data_df_indexed, signal_names = signals, window_size_samples = window_size_samples, metrics=metrics, existing_feature_cols=existing_feature_cols)

            data_df.loc[data_df[ACTIVITY_IDX_COL] == act_index, generated_feature_cols] = data_df_indexed[generated_feature_cols]
        
        # features_df = features_df.dropna()
        # data_df = data_df.dropna()

        features_df_per_w = pd.concat([features_df_per_w, data_df[generated_feature_cols]], axis = 1)
        features_df_per_w = features_df_per_w.dropna()
        assert(len(features_df_per_w) == len(data_df.dropna()))
        # dropna will make sure to drop rows at the beginning
        if INDEX_COL not in features_df_per_w.columns.values:
            features_df_per_w[INDEX_COL] = features_df_per_w.index 

        save_as_pickle(features_df_per_w, activities_features_out_fname_per_w)
        # features_df_per_w.to_csv(activities_features_out_fname_per_w, index = False)
        print(f"Wrote features file {activities_features_out_fname_per_w}")


def generate_train_val_split(input_dir, train_out_dir, val_out_dir, train_split, val_split):
    '''
    For all files in the input_dir generate a train and validation split. 
    '''
    train_split = np.round(train_split, 2)
    val_split = np.round(1.0 - train_split, 2)
    assert((0 < train_split) & (train_split < 1.0))
    assert((0 < val_split) & (val_split < 1.0))

    if not os.path.exists(train_out_dir):
        os.makedirs(train_out_dir)
    if not os.path.exists(val_out_dir):
        os.makedirs(val_out_dir)
    
    files = glob.glob(input_dir + "/AOS*.csv")
    for fname in files: 
        print(f"Entering train val split for {fname}")
        # Define output filenames
        train_out_fname = os.path.join(train_out_dir, os.path.basename(fname).replace("_gt_features", "_gt_features_train"))
        val_out_fname   = os.path.join(val_out_dir, os.path.basename(fname).replace("_gt_features", "_gt_features_val"))

        if os.path.exists(train_out_fname) & os.path.exists(val_out_fname):
            df_features_existing_cols = pd.read_csv(fname, nrows=1, index_col=INDEX_COL).columns.values.tolist()
            try:
                df_features_existing_cols.remove(LABEL_COL)
                df_features_existing_cols.remove(ACTIVITY_IDX_COL)
            except:
                pass
            df_train_existing_cols = pd.read_csv(train_out_fname, nrows=1, index_col=INDEX_COL).columns.values.tolist()
            df_val_existing_cols = pd.read_csv(val_out_fname, nrows = 1, index_col=INDEX_COL).columns.values.tolist()

            if set(df_features_existing_cols).issubset(df_train_existing_cols):
                if set(df_features_existing_cols).issubset(df_val_existing_cols):
                    continue

        # Read the features dataframe
        data_df = pd.read_csv(fname, index_col = INDEX_COL)

        # Get feature columns, i.e. all columns except the ACTIVITY_COL (LABEL_COL) and "ACTIVITY_IDX" column (ACTIVITY_IDX_COL)
        feature_cols_out = list(data_df.columns.values)
        feature_cols_out.remove(LABEL_COL)
        feature_cols_out.remove(ACTIVITY_IDX_COL)

        # Initialize the output dataframes
        train_df = pd.DataFrame(columns=data_df.columns.values)
        val_df = pd.DataFrame(columns=data_df.columns.values)

        for act_index in list(set(list(data_df[ACTIVITY_IDX_COL].values))):
        
            data_df_indexed = data_df.loc[data_df[ACTIVITY_IDX_COL] == act_index]

            cut_idx = int(train_split * len(data_df_indexed))
            train_df    = train_df.append(data_df_indexed.iloc[:cut_idx])
            val_df      = val_df.append(data_df_indexed.iloc[cut_idx:])

        # From here now on we dont need the ACTIVITY_IDX column anymore
        train_df = train_df[[LABEL_COL] + feature_cols_out]
        val_df = val_df[[LABEL_COL] + feature_cols_out]

        # check if INDEX col is train_df and val_df
        if INDEX_COL not in train_df.columns.values:
            train_df[INDEX_COL] = train_df.index
        if INDEX_COL not in val_df.columns.values:
            val_df[INDEX_COL] = val_df.index

        # Write to file
        train_df.to_csv(train_out_fname, index = False)
        print(f"Wrote {train_out_fname}")
        val_df.to_csv(val_out_fname, index = False)
        print(f"Wrote {val_out_fname}")

def generate_normalized_train_val_set(train_dir, val_dir, train_out_dir_normalized, val_out_dir_normalized, scaler = "standard"):
    '''
    :param scaler: scaler to be used. Either "standard" or "min_max"
                    "standard" removes the mean and scales features to have unit variance
                    "min_max" scales the features to values in [min, max], in the current implementation to [0, 1]
    '''
    train_files = glob.glob(train_dir + "/AOS*.csv")
    val_files = [os.path.join(val_dir, os.path.basename(f).replace("_train", "_val")) for f in train_files] # corresponding validation files
     
    if not os.path.exists(train_out_dir_normalized):
        os.makedirs(train_out_dir_normalized)
    if not os.path.exists(val_out_dir_normalized):
        os.makedirs(val_out_dir_normalized)


    for train_file, val_file in zip(train_files, val_files):
        print(f"Entering normalization for train file {train_file} and corresponding val file {val_file}")
        train_out_fname = os.path.join(train_out_dir_normalized, os.path.basename(train_file))
        val_out_fname   = os.path.join(val_out_dir_normalized, os.path.basename(val_file))

        if os.path.exists(train_out_fname) & os.path.exists(val_out_fname):
            continue 
        
        try: 
            train_df = pd.read_csv(train_file, index_col = INDEX_COL)
            val_df = pd.read_csv(val_file, index_col = INDEX_COL)
        except: 
            train_df = pd.read_csv(train_file, index_col = "Unnamed: 0")
            val_df = pd.read_csv(val_file, index_col = "Unnamed: 0")

        feature_cols = list(train_df.columns.values)
        if LABEL_COL in feature_cols:
            feature_cols.remove(LABEL_COL)
        if ACTIVITY_IDX_COL in feature_cols:
            feature_cols.remove(ACTIVITY_IDX_COL)

        if scaler == "standard":
            train_df, val_df = scale_standard(train_df, feature_cols, val_df = val_df)
        elif scaler == "min_max":
            train_df, val_df = scale_min_max(train_df, val_df, feature_cols, val_df = val_df)
        else:
            print(f"No valid feature scaling method {scaler}")
            sys.out()

        # check if INDEX col is train_df and val_df
        if INDEX_COL not in train_df.columns.values:
            train_df[INDEX_COL] = train_df.index
        if INDEX_COL not in val_df.columns.values:
            val_df[INDEX_COL] = val_df.index

        if INDEX_COL + ".1" in train_df.columns.values:
            train_df.drop(INDEX_COL + ".1", inplace = True, axis = 1)
        
        if INDEX_COL + ".1" in val_df.columns.values:
            train_df.drop(INDEX_COL + ".1", inplace = True, axis = 1)

        train_df.to_csv(train_out_fname, index = False)
        print(f"Wrote {train_out_fname}")
        val_df.to_csv(val_out_fname, index = False)
        print(f"Wrote {val_out_fname}")

def combine_features_dirs(features_dirs_w, output_dir, database = "aos"):
    '''
    Combine feature files from the same patient created with different window sizes 
    
    :param features_dirs_w: list of directories containing the patient file, will look for same patient file in all directories. 
    :param output_dir: where to save the combined output feature file

    '''

    for features_dir_w in features_dirs_w:
        if database == "aos":
            filenames_w = [os.path.basename(f) for f in get_aos_files_from_dir(features_dir_w)]
        elif database == "leipzig":
            filenames_w = [os.path.basename(f) for f in get_leipig_files_from_dir(features_dir_w)]
        else: 
            sys.exit()

        for f_basename in filenames_w: 
            combined_out_fname = os.path.join(output_dir, f_basename)
            fname_w = os.path.join(features_dir_w, f_basename)

            if os.path.exists(combined_out_fname):
                # Check if all the features are in the file already
                # df_combined_existing_cols = pd.read_csv(combined_out_fname, nrows=1, index_col=INDEX_COL).columns.values.tolist()
                # df_single_existing_cols = pd.read_csv(fname_w, nrows = 1, index_col=INDEX_COL).columns.values.tolist()
                df_combined_existing_cols = load_from_pickle(combined_out_fname).iloc[:1].columns.values.tolist()
                df_single_existing_cols = load_from_pickle(fname_w).iloc[:1].columns.values.tolist()

                if set(df_single_existing_cols).issubset(df_combined_existing_cols):
                    # Features already in "big" feature file
                    os.remove(fname_w)
                    continue
                combined_features_df = load_from_pickle(combined_out_fname)
                combined_features_df.set_index(INDEX_COL)
            else:
                combined_features_df = load_from_pickle(fname_w)
                combined_features_df.set_index(INDEX_COL)

            df_single_existing_cols = load_from_pickle(fname_w).iloc[:1].columns.values.tolist()

            missing_cols = [c for c in df_single_existing_cols if c not in combined_features_df.columns.values]

            single_features_df = load_from_pickle(fname_w)[missing_cols + [INDEX_COL]]
            single_features_df.set_index(INDEX_COL)
            combined_features_df = merge_dataframes(combined_features_df, single_features_df) 

            if INDEX_COL not in combined_features_df.columns.values:
                combined_features_df[INDEX_COL] = combined_features_df.index

            save_as_pickle(combined_features_df, combined_out_fname)

            # Remove individual files
            # os.remove(fname_w)
    return 

def normalize_feature_dirs(features_dir_in, features_dir_out, scaler = "standard", database = "aos"):
    '''
    For each input feature file in features_dir_in create a feature normalized file.
    
    :param features_dir_in: directory containing the features files
    :param features_dir_out: directory where normalized files should be saved
    '''
    if database == "aos":
        files_in = glob.glob(features_dir_in + "/AOS*.pkl")
    elif database == "leipzig":
        files_in = glob.glob(features_dir_in + "/*[0-9].pkl")
    else:
        print(f"Invalid database {database}")
        sys.exit()
    for f_in in files_in: 
        f_out = os.path.join(features_dir_out, os.path.basename(f_in).replace(".pkl", "_normalized.pkl"))

        feature_cols_in = list(load_from_pickle(f_in).iloc[:1].columns.values)

        if os.path.exists(f_out):
            feature_cols_out = list(load_from_pickle(f_out).iloc[:1].columns.values)
            if set(list(feature_cols_in)).issubset(set(list(feature_cols_out))):
                # normalized feature file already exists
                continue
        
        remove_cols = [LABEL_COL, ACTIVITY_IDX_COL, INDEX_COL]
        for rm_col in remove_cols:
            if rm_col in feature_cols_in:
                feature_cols_in.remove(rm_col)

        # features_df = pd.read_csv(f_in, index_col = INDEX_COL)
        features_df = load_from_pickle(f_in)
        features_df.set_index(INDEX_COL)
        if scaler == "standard":
            features_normalized_df = scale_standard(features_df, feature_cols_in)
        elif scaler == "min_max":
            features_normalized_df = scale_min_max(features_df, feature_cols_in)
        else:
            print(f"No valid feature scaling method {scaler}")
            sys.out()
        
        if INDEX_COL not in features_normalized_df.columns.values:
            features_normalized_df[INDEX_COL] = features_normalized_df.index 

        save_as_pickle(features_normalized_df, f_out) #, index = False)
        print(f"Wrote features file {f_out}")

    return


def generate_features_multiple_window_sizes(input_dir, features_dir, window_sizes, signals, metrics):
    '''
    This method calls the generate_features method multiple times for different window sizes 
    
    :window_size: the window size to use for this 
    :param features_dir: the directory where the "big" feature fill lies. output will be in features_dir/w_size_{window_size}
    :param window_sizes: 
    :param signals: 
    :param metrics:

    :return: return the individual output directories  
    '''
    features_dirs_w = []

    for window_size_samples in window_sizes:

        features_dir_w = os.path.join(features_dir, get_window_size_str(window_size_samples))

        if not os.path.exists(features_dir_w):
            os.makedirs(features_dir_w)

        for signal in signals: 
            # "Batchwise" per feature generation, one could also give the list of signals directory to the generate features method
            print(f"Feature generation for window size {window_size_samples} and signal {signal}.")
            generate_features(input_dir = input_dir, output_dir = features_dir_w, signals = [signal], window_size_samples=window_size_samples, metrics=metrics, features_dir=features_dir)
        
        # generate_features(input_dir = input_dir, output_dir = features_dir_w, signals = [signal], window_size_samples=window_size_samples, metrics=metrics) # this does not work for big files due to computation time
        features_dirs_w.append(features_dir_w)

    return features_dirs_w

def dataset_generation_shallow_main(activities_output_dir = ACTIVITIES_OUTPUT_DIR,
                            features_dir = FEATURES_DIR,
                            features_dir_normalized = FEATURES_NORMALIZED_DIR,
                            window_sizes = WINDOW_SIZES, 
                            signal_cols_features = SIGNAL_COLS_FEATURES,
                            metrics = METRICS,
                            database = DATABASE):
    # # if only normalization is desired uncomment
    activity_code_type_dict = get_dir_activity_code_type_dict(activities_output_dir)
    save_activity_code_type_dict(activity_code_type_dict, features_dir_normalized)
    normalize_feature_dirs(features_dir, features_dir_normalized, database = database)
    sys.exit()
    # Generate Features
    features_dirs_w = generate_features_multiple_window_sizes(input_dir = activities_output_dir, features_dir = features_dir, window_sizes = window_sizes, signals = signal_cols_features, metrics=metrics)
    activity_code_type_dict = get_dir_activity_code_type_dict(activities_output_dir)
    save_activity_code_type_dict(activity_code_type_dict, features_dir)
    combine_features_dirs(features_dirs_w, features_dir, database = database)
    
    # Normalize the features per file
    save_activity_code_type_dict(activity_code_type_dict, features_dir_normalized)
    normalize_feature_dirs(features_dir, features_dir_normalized, database = database)

    return

if __name__ == "__main__":

    dataset_generation_shallow_main()
