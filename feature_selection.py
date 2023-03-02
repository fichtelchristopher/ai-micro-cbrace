''' 
Master thesis project of Christopher Fichtel started in Oct 2022
AI for microprocessor-controlled knee joints

File created 09 Nov 2022

Code Base for selecting the best features.
'''
import numpy as np
import pandas as pd

import glob

import os

from Misc.utils import *

from Configs.pipeline_config import *

from Processing.file_handling import get_train_dir, get_val_dir

from sklearn.feature_selection import mutual_info_classif

from Configs.shallow_pipeline_config import *

from sklearn.decomposition import PCA, IncrementalPCA

'''
Methods: 
mutual info:    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html 
                Mutual information (MI) [1] between two random variables is a non-negative value, which measures the dependency between the variables. It is equal to zero if and only if two random variables are independent, and higher values mean higher dependency.
                Returns estimated mutual information between each feature and the target.
                --> select features with HIGH mutual info i.e. high dependency between feature and class label
# TODO handle "final" feature which is the same for all window sizes
'''

def feature_selection(train_files, feature_selection_method, num_features, feature_selection_per_file = False, loc_indices_list = None):
    '''
    :param train_files: list of training files, train files point to the pkl files containing the features
    :param feature_selection_method
                "mutual_info":   performs feature selection and saves them in self.selected_feature_cols  
                "pca":           performs pca and saves the fitted pca in transformer
    :param num_features: number of feature s to be output
    :param feature selection_per_file
    :param loc_indices_list: if submitted, one list entry per train file
    '''
    assert(feature_selection_method in ["mutual_info", "pca"])
    
    selected_feature_cols = []
    feature_transformer = None
    data_dir = Path(train_files[0]).parent
    # Define the output paths
    if feature_selection_method in ["mutual_info"]:
        feature_scores_out_dir = os.path.join(data_dir, f"feature_scores_{feature_selection_method}")
        if not os.path.exists(feature_scores_out_dir):
            os.makedirs(feature_scores_out_dir)
        if feature_selection_per_file:
            feature_scores_file = os.path.join(feature_scores_out_dir, f"feature_scores_{feature_selection_method}_accum_per_file.pkl")
        else:
            feature_scores_file = os.path.join(feature_scores_out_dir, f"feature_scores_{feature_selection_method}.pkl")
        # Create the feature scores files
        if os.path.exists(feature_scores_file):
            feature_scores_df = load_from_pickle(feature_scores_file)
            if "fname" in feature_scores_df.columns.values:
                feature_scores_df.set_index("fname")
            # Check if all columns of the features files are in the accumulated features dataframe
        
            for idx, f in enumerate(train_files):
                feature_cols = list(load_from_pickle(f).set_index(INDEX_COL).iloc[:1].columns.values)
                if LABEL_COL in feature_cols:
                    feature_cols.remove(LABEL_COL)
                if not set(feature_cols).issubset(set(list(feature_scores_df.columns.values))):
                    feature_scores_df = get_accumulated_feature_scores(train_files, method = feature_selection_method, feature_selection_per_file = feature_selection_per_file, fout = feature_scores_file, loc_indices_list=loc_indices_list)
                    break
        else:
            feature_scores_df = get_accumulated_feature_scores(train_files, method = feature_selection_method, feature_selection_per_file = feature_selection_per_file, fout = feature_scores_file, loc_indices_list=loc_indices_list)
        k_best_features = get_k_best(feature_scores_df, feature_selection_method, num_features)
        selected_feature_cols = k_best_features
    else:
        if feature_selection_method in ["pca"]:
            selected_feature_cols, feature_transformer = get_pca_transformer(train_files, num_features, loc_indices_list=loc_indices_list)
        else:
            print("Invalid feature selector")
            sys.exit()
    
    return selected_feature_cols, feature_transformer


def get_accumulated_feature_scores(files, method = "mutual_info", feature_selection_per_file = True, fout = None, loc_indices_list = None): 
    '''
    per feature get the accumulated feature score based on the method selected #TODO implement methods
    :param feature_selection_per_file: whether each file should be investigated individually and the results then accumulated 
    '''
    # files = glob.glob(train_dir + "/AOS*.csv")

    assert(len(files) > 0)
    
    print(f"\nFeature selection (selection per file = {feature_selection_per_file}) for files: ")

    for c, f in enumerate(files): 
        print(f"Loading {f}.")
        data_df_per_file = load_from_pickle(f).set_index(INDEX_COL)

        if type(loc_indices_list) != type(None):
            if len(loc_indices_list) == len(files):
                data_df_per_file = data_df_per_file.loc[loc_indices_list[c]]

        if feature_selection_per_file:
            if type(fout) != type(None):
                individual_fout = fout.replace("accum_per_file", os.path.basename(f).replace(".pkl", ""))
                individual_fout = individual_fout.replace("_gt", "")
                individual_fout = individual_fout.replace("_features_normalized", "")
            else:
                individual_fout = None
            feature_scores = get_feature_scores(data_df_per_file, method, fout = individual_fout)

            feature_scores["fname"] = [os.path.basename(f)]

            if c == 0:
                feature_scores_accum = feature_scores
            else:
                feature_scores_accum = feature_scores_accum.append(feature_scores)
        else:
            if c == 0: 
                data_df = data_df_per_file
            else:
                data_df = data_df.append(data_df_per_file)
    
    if not feature_selection_per_file: 
        if method in ["mutual_info"]:
            feature_scores_accum = get_feature_scores(data_df, method) 
            feature_scores_accum["fname"] = "all"

    feature_scores_accum = feature_scores_accum.set_index("fname")
    
    if type(fout) != type(None):
        save_as_pickle(feature_scores_accum, fout)
    
    return feature_scores_accum

def get_pca_transformer(files, n_features, drop_cols = [ACTIVITY_IDX_COL, LABEL_COL, INDEX_COL], loc_indices_list = None):
    '''
    Read here
    https://ieeexplore.ieee.org/abstract/document/5581013?casa_token=PvOYDBNifWgAAAAA:umcQ6TEU_MzUefJVhpRJ17hEvxXBsYQI4yA1W12Rmg602PCJoFn-qKhr3Gr8bo4Zu8w8mY_t  
    '''
    print(f"Getting the pca transformation for {len(files)} files and {n_features} principal components.")

    for c, f in enumerate(files): 
        print(f"Loading {f}.")
        data_df_per_file = load_from_pickle(f)
        data_df_per_file.set_index(INDEX_COL)

        if type(loc_indices_list) != type(None):
            if len(loc_indices_list) == len(files):
                data_df_per_file = data_df_per_file.loc[loc_indices_list[c]]

        if c == 0: 
            data_df = data_df_per_file
        else:
            data_df = data_df.append(data_df_per_file)
    
    feature_cols = list(data_df.columns.values)
    for drop_col in drop_cols: 
        if drop_col in feature_cols:
            feature_cols.remove(drop_col)
    
    feature_df = data_df[feature_cols]

    # pca = PCA(n_components=n_features)
    pca = IncrementalPCA(n_components=n_features)

    # Fit the PCA model to the data
    pca.fit(feature_df)
    # you can now call pca.transform(feature_df)

    return feature_cols, pca

def get_feature_scores(data_df, method, label_col = LABEL_COL, drop_cols = [ACTIVITY_IDX_COL],  df_min_factor = 10, fout = None): 
    '''
    :param df_min_factor: select only the "df_min_factor"th value --> reduces amount of data
    :param method: feature selection, so far implemented: "mutual_info"
    :param f_out: output name for features dataframe. Csv file. If not submitted --> dont save 
    '''
    if type(fout) != type(None):
        if os.path.exists(fout):
            feature_scores = pd.read_csv(fout)
            return feature_scores

    data_df = data_df.iloc[::df_min_factor]

    for c in drop_cols: 
        if c in data_df.columns.values:
            data_df.drop(c, inplace = True, axis = 1)

    feature_df = data_df.drop(label_col, axis = 1)
    label_df = data_df[label_col]

    if method == "mutual_info":
        mutual_info = mutual_info_classif(feature_df, label_df)
    else:
        print(f"Unknown method for feature score computaiton {method}.")
        sys.exit()

    cols = feature_df.columns.values
    # Now data_df only has feature columns 
    feature_scores = pd.DataFrame(np.expand_dims(mutual_info, axis = 0), columns = cols)

    if type(fout) != type(None):
        feature_scores.to_csv(fout, index = False)

    return feature_scores

def get_k_best(feature_scores_df, method, k):
    '''
    :param feature_scores_df: a dataframe containing 
    :param method: the selected feature selection method
    :param k: the number of features to be used
    :return: list of the k best features indicated (list of strings - each representing a column name)
    '''
    feature_cols = np.array(feature_scores_df.columns.values)
    feature_scores_df = np.array(feature_scores_df.sum())

    if method in ["mutual_info"]:
        reverse = True
    else:
        reverse = False

    feature_scores_df, feature_cols = zip(*sorted(zip(feature_scores_df, feature_cols), reverse = reverse))

    feature_cols_k = feature_cols[:k]

    return feature_cols_k

def get_k_best_from_file(f, method, k):
    '''
    '''
    feature_scores = load_from_pickle(f)
    if "fname" in feature_scores.columns.values:
        feature_scores.set_indx("fname")

    return get_k_best(feature_scores, method, k)


"""
def feature_selection_main(train_val_dir = TRAIN_VAL_NORMALIZED_DIR,
                            train_split = TRAIN_SPLIT,
                            selection_per_file = SELECTION_PER_FILE,
                            num_features = NUM_FEATURES,
                            split_mode = SPLIT_MODE,
                            features_normalized_dir = FEATURES_NORMALIZED_DIR,
                            method = "mutual_info"):
    '''
    :param selection_per_file: 
    '''
    if split_mode == "standard":
        data_dir = get_train_dir(train_val_dir, train_split)
    elif split_mode == "loso":
        data_dir = features_normalized_dir
    else:
        print(f"Invalid split mode {split_mode}") 
    
    # Define the output paths
    feature_scores_out_dir = os.path.join(data_dir, f"feature_scores_{method}")
    if not os.path.exists(feature_scores_out_dir):
        os.makedirs(feature_scores_out_dir)
    
    if selection_per_file:
        feature_scores_file = os.path.join(feature_scores_out_dir, f"feature_scores_{method}_accum_per_file.csv")
        k_best_features_file = os.path.join(feature_scores_out_dir, f"{num_features}_best_features_{method}_accum_per_file.npy")
    else:
        feature_scores_file = os.path.join(feature_scores_out_dir, f"feature_scores_{method}.csv")
        k_best_features_file = os.path.join(feature_scores_out_dir, f"{num_features}_best_features_{method}.npy") 

    # Create the feature scores files
    if os.path.exists(feature_scores_file):
        feature_scores_df = pd.read_csv(feature_scores_file, index_col = "fname")
        # Check if all columns of the features files are in the accumulated features dataframe
        for f in glob.glob(feature_scores_out_dir + "/AOS*gt_features_train.csv"):
            feature_cols = list(pd.read_csv(f, nrows = 1, index_col=INDEX_COL).columns.values)
            feature_cols.remove(LABEL_COL)
            if not set(feature_cols).issubset(set(list(feature_scores_df.columns.values))):
                feature_scores_df = get_accumulated_feature_scores(data_dir, method = method, selection_per_file = selection_per_file, fout = feature_scores_file)
    else:
        feature_scores_df = get_accumulated_feature_scores(data_dir, method = method, selection_per_file = selection_per_file, fout = feature_scores_file)

    k_best_features = get_k_best(feature_scores_df, method, num_features)

    np.save(k_best_features_file, k_best_features)
    print(k_best_features)

    return k_best_features


if __name__ == "__main__":
    feature_selection_main()
"""