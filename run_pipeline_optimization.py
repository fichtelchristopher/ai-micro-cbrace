''' 
Master thesis project of Christopher Fichtel started in Oct 2022
AI in operating ortheses

File created 23 Dec 2022

This file comprises the steps for activity classification for both the 
feature based and neural network approach.
'''
from Configs.pipeline_config import * 
from Configs.nn_pipeline_config import *
from Configs.shallow_pipeline_config import *
from classifier import *
from Models.neural_networks import *
from Models.neural_networks_configurations import *
import glob
from Processing.file_handling import get_model_basename_from_leave_out_test_file
from datetime import datetime

from Processing.loading_saving import *

from Misc.output import *
from optimization import *
from Processing.loading_saving import get_leipig_files_from_dir, get_aos_files_from_dir
import optuna
from optuna.samplers import TPESampler

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def run_pipeline_optimization(classifier_type, classifier_config, pipeline_config, mode = "loso", clean_up_f1_threshold = 0.65):
    # The files to be tested
    input_files = get_train_test_files(classifier_type)

    assert(len(input_files) > 0)
    assert(pipeline_config["TRAIN_VAL_SPLIT"]) < 1.0

    basename = pipeline_config["MODEL"] + f"_{len(input_files)}_train_files"
    
    try:
    # if True:
        classifier = get_classifier(classifier_type, classifier_config, pipeline_config, basename)
        history = classifier.train(input_files)
        return  history, classifier.classifier_output_dir
    except: 
    # else:
        return None, ""

def run_pipeline_objective(trial, classifier_type, classifier_config, pipeline_config, output_dir, mode = "loso"):

    optimization_progress_output_f = os.path.join(output_dir, f"study_optimization_progress.txt")
    trial_results_f = os.path.join(output_dir, f"study_optimization_trial_results.csv")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    classifier_config = trial_suggest_classifier_configuration(trial, classifier_type, classifier_config)
    pipeline_config = trial_suggest_pipeline_configuration(trial, classifier_type, pipeline_config)

    # pipeline_config["STEP_SIZE_TEST"] = 25

    print(f"Running optimization pipeline for {classifier_type} with parameters: {trial.params}")

    history, classifier_output_dir = run_pipeline_optimization(classifier_type, classifier_config, pipeline_config, mode = mode)

    if type(history) != type(None):
        accuracy =  history["val_accuracy"] #if "val_acc" in history.keys() else history["accuracy"]
        loss = history["val_loss"] #if "val_loss" in history.keys() else history["loss"]
        f1_score = history["val_f1"] #if "val_f1" in history.keys() else history["f1"]

        print_optimization_progress(trial, optimization_progress_output_f, history)
        save_trial_results(trial, trial_results_f, history, classifier_output_dir) #, classifier_config, pipeline_config)

        return loss
    else:
        return 100

def optimize(classifier_type, n_trials, mode = "loso"):

    sampler = TPESampler(seed=666)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    pipeline_config = get_default_pipeline_config(classifier_type)
    classifier_config = get_default_classifier_config(classifier_type)

    # Output file 
    output_dir = os.path.join(PARAMETER_OPTIMIZATION_DIR, classifier_type)
    study_str = f"study_{str(datetime.now())}"

    study_str = study_str.replace(":", "-")
    study_str = study_str.replace(".", "-")
    output_dir = output_dir + f"/{study_str}"

    objective_func = lambda trial: run_pipeline_objective(trial, classifier_type, classifier_config, pipeline_config, output_dir, mode)

    study.optimize(objective_func, n_trials=n_trials)
    return study, output_dir

if __name__ == "__main__":

    # BASIC OPTIMIZATION PARAMETER
    classifier_type = "rf"
    n_trials = 100

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Val mode is the faster optionm. asserts that the pipeline config has a train val split value > 0 and < 1
    mode = "val" #valid modes are "loso" "val" 
    
    # Run optimization
    print(f"Starting optimization for {classifier_type} with {n_trials} trials.")
    study, output_dir = optimize(classifier_type, n_trials, mode = mode)
    best_params = study.best_params

    # Save the best parameters as a pickle file
    best_params_output_file = os.path.join(output_dir, f"study_{n_trials}_best_parameters.pkl")
    with open(best_params_output_file, 'wb') as handle:
        pickle.dump(best_params, handle, protocol=pickle.HIGHEST_PROTOCOL)