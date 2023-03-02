from Configs.pipeline_config import * 
from Configs.nn_pipeline_config import *
import sys
import numpy as np
import pandas as pd
 
def trial_suggest_classifier_configuration(trial, classifier_type, classifier_configuration):
    # for all NN implementations --> batch normalizatoin yes or no
    if classifier_type in ["LSTM", "CNN", "CNN_LSTM"]:
        classifier_configuration["batch_normalization"] = trial.suggest_categorical("batch_normalization", [True, False])

    if classifier_type == "LSTM":
        # architecture suggestion by trial for CNN
        classifier_configuration["lstm_layers_number"] = trial.suggest_int("lstm_layers_number", 1, 3)

        for i in range(classifier_configuration["lstm_layers_number"]):
            classifier_configuration[f"n_filters_lstm_{i}"] = int(trial.suggest_int(f'n_filters_lstm_{i}', 2, 101, step = 2))

        # classifier_configuration["dropout"] = trial.suggest_float("dropout", 0.3, 0.51, step = 0.1)

        classifier_configuration["dense_layers_number"] = trial.suggest_int("dense_layers_number", 1, 3)
        for i in range(classifier_configuration["dense_layers_number"]):
            classifier_configuration[f'n_filters_dense_{i}'] = int(trial.suggest_categorical(f'n_filters_dense_{i}', [2, 4, 8, 16, 32, 64, 128, 256]))

    elif classifier_type == "CNN":
        # architecture suggestion by trial for CNN
        classifier_configuration["layers_number"]       = int(trial.suggest_int("layers_number", 1, 3))
        classifier_configuration["conv_filter_size"]    = int(trial.suggest_categorical("conv_filter_size", [2, 4, 8]))
        classifier_configuration["pool_size"]           = int(trial.suggest_categorical("pool_size", [2, 4]))

        for i in range(classifier_configuration["layers_number"]):
            classifier_configuration[f"n_filters_conv_{i}"] = int(trial.suggest_categorical(f"n_filters_conv_{i}", [2, 4, 16, 64, 256]))

        classifier_configuration[f"n_filters_dense"] = int(trial.suggest_categorical("n_filters_dense", [2, 4, 8, 16, 32, 64, 128, 256]))

        classifier_configuration[f"dropout"] = trial.suggest_float("dropout", 0.3, 0.51, step = 0.1)

    elif classifier_type == "CNN_LSTM":
        # architecture suggestion by trial for CNN_LSTM
        classifier_configuration[f"conv_layers_number"] = trial.suggest_int("conv_layers_number", 1, 3)
        classifier_configuration[f"conv_filter_size"]   = int(trial.suggest_categorical("conv_filter_size", [2, 4, 8]))
        classifier_configuration["pool_size"]           = int(trial.suggest_categorical("pool_size", [2, 4]))

        for i in range(classifier_configuration[f"conv_layers_number"]):
            classifier_configuration[f"n_filters_conv_{i}"] = int(trial.suggest_categorical(f'n_filters_conv_{i}', [2, 4, 8, 16, 32, 64, 128, 256]))

        classifier_configuration[f"lstm_layers_number"] = trial.suggest_int("lstm_layers_number", 1, 3)

        for i in range(classifier_configuration[f"lstm_layers_number"]):
            classifier_configuration[f'n_filters_lstm_{i}'] = int(trial.suggest_int(f'n_filters_lstm_{i}', 2, 101, step = 2))

        classifier_configuration[f"dropout"] = trial.suggest_float("dropout", 0.3, 0.51, step = 0.1)
    
            
    elif classifier_type == "LSTM_SIMPLE":
        classifier_configuration["lstm_units"] = int(trial.suggest_categorical("lstm_units", [32, 64, 128, 256]))
       
        classifier_configuration["dropout"] =  trial.suggest_float("dropout", 0.1, 0.4, step = 0.1)
        # classifier_configuration["activation"] = trial.suggest_categorical("activation", ["softmax", 'sigmoid'])  

    elif classifier_type == "CNN_SIMPLE":

        classifier_configuration["n_layers_conv"] =  int(trial.suggest_categorical("n_layers_conv", [1, 2]))
        classifier_configuration["n_filters_conv"] = int(trial.suggest_categorical("n_filters_conv", [32, 64, 128]))
        classifier_configuration["kernel_size_conv"] = int(trial.suggest_categorical("kernel_size_conv", [3, 5, 7]))

        classifier_configuration["pool_size"] = int(trial.suggest_categorical("pool_size", [2]))#, 3, 4]))
        classifier_configuration["pool_type"] = trial.suggest_categorical("pool_type", ["max", "avg"])


        classifier_configuration["dense_units"] = int(trial.suggest_categorical("dense_units", [32, 64, 128]))

        classifier_configuration["dropout"] =  trial.suggest_float("dropout", 0.3, 0.5, step = 0.1)
        # classifier_configuration["activation"] = trial.suggest_categorical("activation", ["softmax", 'sigmoid'])  
        
    elif classifier_type == "CNN_LSTM_SIMPLE":
        classifier_configuration["n_layers_conv"] =  int(trial.suggest_categorical("n_layers_conv", [1, 2]))
        classifier_configuration["n_filters_conv"] = int(trial.suggest_categorical("n_filters_conv", [32, 64, 128]))
        classifier_configuration["kernel_size_conv"] = int(trial.suggest_categorical("kernel_size_conv", [3, 5, 7]))

        classifier_configuration["pool_size"] = int(trial.suggest_categorical("pool_size", [2, 3, 4]))
        classifier_configuration["pool_type"] = trial.suggest_categorical("pool_type", ["max", "avg"])

        classifier_configuration["lstm_units"] = int(trial.suggest_categorical("lstm_units", [32, 64, 128]))

        classifier_configuration["conv_dropout"] =  trial.suggest_float("dropout", 0.2, 0.4, step = 0.1)
        classifier_configuration["lstm_dropout"] =  trial.suggest_float("dropout", 0.1, 0.3, step = 0.1)

    elif classifier_type.lower() == "svm":
        # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        classifier_configuration["kernel"] = str(trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"]))
        classifier_configuration["gamma"] = str(trial.suggest_categorical("gamma", ["scale", "auto"]))
        classifier_configuration["shrinking"] =  trial.suggest_categorical("shrinking", [True, False])
        classifier_configuration["tol"] = float(trial.suggest_categorical("tol", [0.01, 0.001, 0.0001]))
        classifier_configuration["class_weight"] = "balanced"
        classifier_configuration["decision_function_shape"] = str(trial.suggest_categorical("decision_function_shape", ["ovo", "ovr"]))
    
    elif classifier_type.lower() == "sgd":
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
        classifier_configuration["loss"] = str(trial.suggest_categorical("loss", ["hinge", "log_loss", "modified_huber", "perceptron"]))
        classifier_configuration["penalty"] = str(trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]))
        classifier_configuration["alpha"] = float(trial.suggest_categorical("alpha", [0.1, 0.001, 0.0001, 0.00001]))
        classifier_configuration["max_iter"] = int(trial.suggest_categorical("max_iter", [100, 250, 1000, 1500]))
        classifier_configuration["tol"] = float(trial.suggest_categorical("tol", [0.01, 0.001, 0.0001]))

    elif classifier_type.lower() == "rf":
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        classifier_configuration["n_estimators"] =  int(trial.suggest_int("n_estimators", 10, 150, 10)) #aka learning rate
        classifier_configuration["criterion"] = str(trial.suggest_categorical("criterion", ["gini", "entropy"]))
        classifier_configuration["max_depth"] = trial.suggest_categorical("max_depth", [None, 2, 3, 5, 7, 9])
        classifier_configuration["min_samples_split"] =  int(trial.suggest_int("min_samples_split", 2, 6))
        classifier_configuration["min_samples_leaf"] =  int(trial.suggest_int("min_samples_leaf", 1, 4))
        classifier_configuration["min_weight_fraction_leaf"] =  float(trial.suggest_float("min_weight_fraction_leaf", 0, 0.5, step = 0.1))
        classifier_configuration["max_features"] =  trial.suggest_categorical("max_features", ["sqrt", "log2", None])
        classifier_configuration["oob_score"] =  trial.suggest_categorical("oob_score", [True, False])
        classifier_configuration["class_weight"] =  trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None])
    
    elif classifier_type.lower() == "xgboost":
        # https://xgboost.readthedocs.io/en/stable/parameter.html#global-config

        # General parameters
        classifier_configuration["booster"] = str(trial.suggest_categorical("booster", ["gbtree", "dart"]))
        classifier_configuration["validate_parameters"] =  bool(trial.suggest_categorical("validate_parameters", [True, False]))

        # Tree booster parameters
        classifier_configuration["eta"] =  float(trial.suggest_float("eta", 0.1, 1.0, step = 0.1)) #aka learning rate
        classifier_configuration["gamma"] =  float(trial.suggest_float("gamma", 0, 1.0, step = 0.01)) # larger gamma , more conservative, minimum loss reduction required, default zero
        classifier_configuration["max_depth"] = int(trial.suggest_int("max_depth", 1, 9)) # default 6
        classifier_configuration["min_child_weight"] = float(trial.suggest_float("min_child_weight", 0, 2.0, step = 0.1)) #default 1
        classifier_configuration["max_delta_step"] = float(trial.suggest_float("max_delta_step", 0, 1.0, step = 0.1)) #default 1
        classifier_configuration["subsample"] = float(trial.suggest_float("subsample", 0, 1.0, step = 0.1)) #default 1
        classifier_configuration["lambda"] = float(trial.suggest_float("lambda", 1, 10, step = 1)) #default 1, l2 regularizaiton
        classifier_configuration["alpha"] = float(trial.suggest_float("alpha", 0, 1.0, step = 0.01)) #default 0, l1 regularizaiton
    
    elif classifier_type.lower() == "lda":
        classifier_configuration["solver"] = str(trial.suggest_categorical("solver", ["svd", "lsqr", "eigen"]))
        classifier_configuration["priors"] = trial.suggest_categorical("priors", [None]) # TODO might implement
        classifier_configuration["n_components"] = trial.suggest_categorical("n_components", [None]) #
        classifier_configuration["tol"] = float(trial.suggest_categorical("tol", [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5]))
    else:
        print(f"Invalid classifier type {classifier_type}")
        sys.exit()


    return classifier_configuration

def trial_suggest_pipeline_configuration(trial, classifier_type, pipeline_configuration):

    if classifier_type in NN_IMPLEMENTATIONS:
        pipeline_configuration["WINDOW_SIZE"] = int(trial.suggest_categorical("WINDOW_SIZE", WINDOW_SIZES_OPTIMIZATION))
        pipeline_configuration["EPOCHS"] = trial.suggest_int("EPOCHS", EPOCH_MIN, EPOCH_MAX, step = EPOCH_STEP)

        pipeline_configuration["LR"] = float(trial.suggest_categorical("LR", INITIAL_LEARNING_RATES))
        pipeline_configuration["LR_DECAY"] = trial.suggest_float("LR_DECAY", LR_DECAY_MIN, LR_DECAY_MAX, step = LR_DECAY_STEP)
        pipeline_configuration["LR_DECAY_AFTER_EPOCHS"] = int(trial.suggest_categorical("LR_DECAY_AFTER_EPOCHS", LR_DECAY_AFTER_EPOCHS))

        pipeline_configuration["BATCH_SIZE"] = int(trial.suggest_categorical("BATCH_SIZE", BATCH_SIZES_OPTIMIZATION))

        # pipeline_configuration["REGULARIZATION_STRENGTH"] = float(trial.suggest_categorical("REGULARIZATION_STRENGTH", REGULARIZATION_STRENGTHS_OPTIMIZATION))

        # pipeline_configuration["CLASS_WEIGHT"][0] = float(trial.suggest_float("ZERO_CLASS_WEIGHT", CLASS_WEIGHT_ZERO_MIN, 1.01, step = 0.1)) # class weigth for the other class
    else:
        pipeline_configuration["NUM_FEATURES"] = int(trial.suggest_categorical("NUM_FEATURES", [5, 10, 15, 25, 50, 75, 100]))
        if DATABASE == "aos":
            pipeline_configuration["TRAIN_PER_FILE"] = True
        else:
            pipeline_configuration["TRAIN_PER_FILE"] = bool(trial.suggest_categorical("TRAIN_PER_FILE", [True, False])) # --> TODO implement
        pipeline_configuration["FEATURE_SELECTION_METHOD"] = trial.suggest_categorical("FEATURE_SELECTION_METHOD", ["pca", "mutual_info"])
    
    pipeline_configuration["MODEL"] = classifier_type

    return pipeline_configuration


def print_optimization_progress(trial, output_f, history_dict):
    '''
    print the currently best parameters to the output file.
    :param trial: an optuna.trial._trial.Trial object, has the .study attribute which contains the currently best parameters
    :param output_file: textfile to write to
    :param f1_score_current_trial: the f1 score of the current trial
    '''
    if os.path.exists(output_f):
        if trial.number == 0:
            os.remove(output_f)

    if trial.number > 0:
        original_stdout = sys.stdout
        with open(output_f, "w") as f:
            sys.stdout = f # Change the standard output to the file we created.
            print(f"\nDone {trial.number} trials.")
            print(f"Parameters at this trial yielded")
            for k, v in history_dict.items():
                print(f"{k} = {v}")
            print("with parameters:")
            print(f"{trial.params}\n")
            print(f"Best parameters so far at trial number {trial.study.best_trial.number} with loss values {trial.study.best_trial.value}:")
            print(f"{trial.study.best_params}\n\n")

            sys.stdout = original_stdout

    return

def save_trial_results(trial, trial_results_f, history, classifier_output_dir):
    '''
    Save the trial configurations and the resulting f1 score
    ''' 

    if os.path.exists(trial_results_f):
        trial_results_df = pd.read_csv(trial_results_f)
    else:
        trial_results_df = pd.DataFrame()
    
    trial_dict = dict(trial.params)

    # trial_dict["f1-score"] = f1_scores_mean
    for k, v in history.items():
        trial_dict[k] = v

    trial_dict["trial_number"] = trial.number
    trial_dict["output_dir"] = classifier_output_dir
    trial_params = pd.DataFrame(trial_dict, index = [trial.number])

    result_df = trial_results_df.append(trial_params, ignore_index = True)

    result_df.to_csv(trial_results_f, index = False)

    return 

