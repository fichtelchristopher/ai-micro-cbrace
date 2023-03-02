''' 
Master thesis project of Christopher Fichtel started in Oct 2022
AI for microprocessor-controlled knee joints

File created 09 Nov 2022

Code Base for training classifiers.
'''

import numpy as np
import pandas as pd

import glob
import os

from Misc.utils import *
from Processing.loading_saving import *
from Processing.file_handling import * 
from Configs.pipeline_config import *
from Configs.shallow_pipeline_config import *

from Processing.loading_saving import *

from Models.neural_networks import * 
from Models.shallow_classifier import * 

from Misc.dataset_utils import *
from Misc.ai_utils import *

from Misc.feature_extraction import *

from Models.neural_networks import * 

from Configs.nn_pipeline_config import *

import pickle

from time import process_time

def fit_model(X_train, Y_train, model_type = "svm", classifier_config = None): 
    ''' 
    :param X_train: the training data in shape [num_samples, num_features]
    :param Y_train: the training labels in shape [num_samples, 1]
    :model_type: one of "svm" (Support Vector Machines) "rf" (Random Forest), "xgboost" or "sgd" (Stochastic Gradient Descent)
    :param classifier_config: see Models.feature_classifier_configuraiton
    :return: fitted model 
    '''

    if model_type.lower() == "svm": 
        model = get_svm(classifier_config)
    elif model_type.lower() == "rf":
        # not working yet #TODO
        model = get_rf(classifier_config)
    elif model_type.lower() == "sgd":
        model = get_sgd(classifier_config)
    elif model_type.lower() == "xgboost":
        model = get_xgboost(classifier_config)
    elif model_type.lower() == "lda":
        model = get_lda(classifier_config)
    elif model_type.lower() == "kmeans":
        model = get_kmeans(n_classes = 3)
    else:
        print(f"Invalid model type {model_type}")

    # for xgboost labels have to be [0, 1, 2, 3, ...] and not [0, 100, 200, ..]
    if model_type.lower() in ["xgboost"]:
        le = get_label_encoder(labels=list(set(list(Y_train))))
        Y_train = le.transform(Y_train)

    if model_type.lower() in ["kmeans"]:
        model.fit(X_train)
    else:
        model.fit(X_train, Y_train)

    # Set model classes to orgiinal format --> inverse transform with the label encoder
    if model_type.lower() in ["xgboost"]:
        model.classes_ = le.inverse_transform(list(model.classes_))

    return model 

def get_features():
    '''
    ''' 
    features_list = []

    return features_list

def verify_feature_availability(files, features_list):
    '''
    Make sure the

    :param files: list of filepaths to feature csv-files.
    :param features_list: the list of features used for training  
    '''

    for f in files:
        existing_cols = pd.read_csv(f, nrows=1, index_col=INDEX_COL).columns.values.tolist()
        if not set(list(features_list)).issubset(set(list(existing_cols))):
            return False
    return True

def fit_classifier(X_train, Y_train, model_type, model_out_fname, output_specs_file = None, classifier_config = None):
    '''
    Run the training. 
    Prints run time to output_specs_file if submitted
    '''
    if os.path.exists(model_out_fname):
        return

    training_start_time = process_time()
    model = fit_model(X_train, Y_train, model_type = model_type, classifier_config = classifier_config)
    training_stop_time = process_time() 
    if type(output_specs_file) != type(None):
        if os.path.isfile(output_specs_file):
            print_to_file(output_specs_file, (f"Fitting {int(len(X_train))} samples took  {training_stop_time - training_start_time} seconds."))
    pickle.dump(model, open(model_out_fname, 'wb'))
    return model

'''
def training_main(  selected_feature_cols = [],
                    window_sizes_samples = WINDOW_SIZES,
                    window_size_samples = WINDOW_SIZE,
                    window_step_size_samples = STEP_SIZE,
                    signal_cols_features = SIGNAL_COLS_FEATURES,
                    metrics = METRICS,
                    train_split = TRAIN_SPLIT,
                    features_normalized_dir = FEATURES_NORMALIZED_DIR,
                    val_split = VAL_SPLIT,
                    model_type = MODEL_TYPE,
                    fixed_window = FIXED_WINDOW,
                    train_per_file = TRAIN_PER_FILE,
                    models_out_dir = MODELS_DIR,
                    train_val_normalized_dir = TRAIN_VAL_NORMALIZED_DIR,
                    split_mode = SPLIT_MODE):
    
    # :params window_sizes_samples/window_size_samples: only important if selected_feature_cols is None 
    # :param train_per_file: 
    

    if fixed_window:
        window_sizes_samples = [window_size_samples]

    if len(selected_feature_cols) == 0:
        # Get all feature cols as defaults
         selected_feature_cols = get_output_feature_cols(signal_cols_features, window_sizes_samples, metrics)
        
    models_out_dir = get_models_predictions_evaluation_out_dir(models_out_dir, model_type, window_step_size_samples, len(selected_feature_cols), fixed_window=fixed_window, window_size_samples=window_size_samples)

    if not os.path.exists(models_out_dir):
        os.makedirs(models_out_dir)

    if split_mode == "loso":
        feature_files = glob.glob(features_normalized_dir + "/AOS*.csv")
    elif split_mode == "standard":
        train_dir = get_train_dir(train_val_normalized_dir, train_split)
        val_dir  = get_val_dir(train_val_normalized_dir, val_split) #  the corresponding validation directory

        feature_files = glob.glob(train_dir + "/AOS*.csv")
        val_files = glob.glob(val_dir + "/AOS*.csv") # only to check feature availability

        # Make sure that the selected feature list is available in every train file
        assert(verify_feature_availability(feature_files, selected_feature_cols) & verify_feature_availability(val_files, selected_feature_cols))
    else:
        print(f"Invalid split mode {split_mode}")

    output_specs_file = os.path.join(models_out_dir, "run_information.txt")
    features_list_file = os.path.join(models_out_dir, "features.txt")

    print_platform_specifications(output_specs_file)
    print_used_features(output_specs_file, selected_feature_cols)
    print_used_features(features_list_file, selected_feature_cols)

    training_start_time = process_time()
    for c, file in enumerate(feature_files):
        model_out_fname = get_model_out_name(models_out_dir, file, train_per_file)
        
        if os.path.exists(model_out_fname):
                continue
        
        print(f"Loading data from {file}")
        
        if train_per_file:
            X_train, Y_train = get_data_labels_from_file(file, selected_feature_cols = selected_feature_cols, is_training = True)
            fit_classifier(X_train, Y_train, model_type, window_step_size_samples, model_out_fname, output_specs_file = output_specs_file)
        else:
            train_files = [f for f in feature_files if f != file]
            X_train, Y_train = get_data_labels_from_files(train_files, selected_feature_cols = selected_feature_cols, is_training = True)
            fit_classifier(X_train, Y_train, model_type, window_step_size_samples, model_out_fname)
        
    training_stop_time = process_time()
    # Create output information
    print_to_file(output_specs_file, (f"\nTraining for the following files took a total of {training_stop_time - training_start_time} seconds using a window step size of {window_step_size_samples}."))
    for f in feature_files:
        print_to_file(output_specs_file, f) 

if __name__ == "__main__":
    training_main()
'''

'''
# NN APPROACH 
Master thesis project of Christopher Fichtel started in Oct 2022
AI in operating ortheses

File created 09 Nov 2022

Code Base for creating the time series predictions

TUTORIALS: 
https://towardsdatascience.com/time-series-prediction-with-lstm-in-tensorflow-42104db39340 


import tensorflow as tf
tf.random.set_seed(7)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import re
from Misc.utils import *
from Processing.loading import *
from AOS_SD_analysis.AOS_Subject import * 
from Processing.preprocessing import *
from Processing.file_handling import * 
from AOS_SD_analysis.AOS_Rule_Set_Parser import * 
from Configs.pipeline_config import *
from Configs.namespaces import *
from Configs.nn_config import *

import pickle

from keras.utils import to_categorical

from time import process_time

from Models.neural_networks import * 
from Misc.dataset_utils import *
from Misc.ai_utils import *

def training_main_nn(activities_dir = ACTIVITIES_OUTPUT_DIR,
        models_dir = MODELS_DIR,
        model_type = NN_CONFIG["ARCHITECTURE"], 
        signal_cols = SIGNAL_COLS,
        label_col = LABEL_COL,
        window_size = NN_CONFIG["WINDOW_SIZE"],
        step_size = NN_CONFIG["STEP_SIZE"],
        epochs = NN_CONFIG["EPOCHS"],
        batch_size = NN_CONFIG["BATCH_SIZE"],
        shuffle_buffer = NN_CONFIG["SHUFFLE_BUFFER"],
        activity_dict = ACTIVITY_CODE_TYPE_DICT_CHRIS):
        

    n_classes = len(list(activity_dict.keys())) # all possible activities
    input_shape = (window_size, len(signal_cols))

    # activities_dir = os.path.join(activities_dir, model_type)
    files = glob.glob(activities_dir + "/AOS*.csv")

    if len(files) == 0:
        print(f"No files in {activities_dir}.")
        sys.exit()


    models_out_dir = get_nn_models_predictions_evaluation_out_dir(models_dir, model_type, window_size, step_size,  epochs)
    if not os.path.exists(models_out_dir):
        os.makedirs(models_out_dir)
    
    for val_file in files[:1]:
        
        # LOSO method --> train files are all available files except the val file
        train_files = [f for f in files if f != val_file]
       
        model_out_fname = get_nn_model_out_name(models_out_dir, val_file, False)
        if os.path.exists(model_out_fname): 
            print(f"Model already exists: {model_out_fname}")
            continue
            
        # Create the dataset and start training                                
        train_dataset = tf.data.Dataset.from_generator(tf_data_generator, args= [train_files, signal_cols, label_col, window_size, step_size, True, batch_size, shuffle_buffer], output_types = (tf.float32, tf.float32),
                                                output_shapes = ((None, window_size, len(signal_cols)),(None, n_classes)))
        model = get_compiled_model(input_shape = input_shape, n_classes_out=n_classes, model_type = model_type)
        model = train_model(model, train_dataset, epochs, early_stopping_acc_threshold = 0.99)
        model.save(model_out_fname)
        print(f"Saved model {model_out_fname}")

        visualise_training(model, model_out_fname)
'''