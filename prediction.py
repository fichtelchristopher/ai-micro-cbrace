''' 
Master thesis project of Christopher Fichtel started in Oct 2022
AI for microprocessor-controlled knee joints

File created 09 Nov 2022

Code Base for making predictions.
'''
import numpy as np
import pandas as pd
import glob
from Misc.utils import *
from Processing.loading_saving import *
from Processing.preprocessing import *
from Processing.file_handling import * 
from Configs.shallow_pipeline_config import *
from Configs.pipeline_config import *
from training import *
from Processing.loading_saving import *
from Misc.dataset_utils import *
from Misc.ai_utils import *
from Misc.feature_extraction import *
from Models.neural_networks import * 
import numpy as np
from AOS_SD_analysis.AOS_Subject import * 
from AOS_SD_analysis.AOS_Rule_Set_Parser import * 
from Configs.namespaces import *
from Configs.nn_pipeline_config import *
from Models.neural_networks import * 


def get_prediction_nn(model, dataset, activity_dict = ACTIVITY_CODE_TYPE_DICT_CHRIS):
    '''
    
    '''
    probabilities = model.predict(dataset)

    one_hot_encoder = get_one_hot_encoder(activity_dict = activity_dict)

    prediction = one_hot_encoder.inverse_transform(probabilities)[:, 0]

    return prediction, probabilities

def get_probabilities_nn(model, dataset):
    '''
    :param dataset: a tf dataset generator object
    '''
    probabilities = model.predict(dataset)

    return probabilities

def combine_predictions_gt(test_file, predictions, window_size_samples):
    '''
    The test dataframe may be split by an Activity Idx. This method 
    creates a dataframe with predictions values at the indices of the 
    original dataframe 
    '''
    if test_file.endswith(".csv"):
        data_df = pd.read_csv(test_file, index_col= INDEX_COL)
    else:
        data_df = load_from_pickle(test_file)
        data_df.set_index(INDEX_COL)

    indices = []
    gt = []
    for activity_idx in list(set(data_df[ACTIVITY_IDX_COL].values)):
        data_df_indexed = data_df[data_df[ACTIVITY_IDX_COL] == activity_idx].iloc[window_size_samples:] # the predicitons first start at window_size_samples
        
        gt.extend(data_df_indexed[LABEL_COL].values)
        indices.extend(data_df_indexed.index.values)

    assert(len(predictions) == len(indices))
    prediction_df = pd.DataFrame(data = predictions, columns=[PREDICTION_COL]) 

    prediction_df[LABEL_COL] = gt

    prediction_df[INDEX_COL] = indices

    return prediction_df
   
def combine_probability_predictions(input_dir, out_fname, dataset_size_dict = None):
    ''' 
    :param input_dir: path to directory that contains model probabilities :para out_fname: where to save the combined classifications


    :param dataset_size_ratios: for when using TRAIN_PER_FILE, the dataset size ratios may be used for computing the accumulated probabilities
                            the idea is that a classifier that has been trained on a higher amount of training data will be assigned a higher importance
                            key     --> the classifier name; the probability file has the same name but "_model_probabilities.pkl" instaed of ".pkl" in the classifier name
                            value:  --> the number of training instances  
    '''
    probabilities_files = glob.glob(input_dir + "/*_probabilities.pkl")
    probabilities_files = [str(Path(f)) for f in probabilities_files]

    dataset_importance_dict = {}
    # Create the importance dict --> ratio of training instances of a classifier and total instances
    if type(dataset_size_dict) != type(None):

        for k, v in dataset_size_dict.items():
            prob_file_mod = str(Path(k.replace(".pkl", "_probabilities.pkl")))

            for f in probabilities_files:
                if os.path.basename(prob_file_mod) == os.path.basename(f): 

                    dataset_importance_dict[f] = v
                    break

        total_instances = sum(dataset_importance_dict.values())
        for k, v in dataset_importance_dict.items():
            dataset_importance_dict[k] = v / total_instances 
    # Make sure that every used probability file is in the dictionary
    if not np.all(np.array(probabilities_files) in np.array(list(dataset_importance_dict.keys()))):
        dataset_importance_dict = None

    probs_list = []
    dataset_importances = []
    for f in probabilities_files: 
        # probs = pd.read_csv(f, index_col= INDEX_COL) 
        probs = load_from_pickle(f).set_index(INDEX_COL)

        probs_list.append(probs)
        if type(dataset_importance_dict) != type(None):
            dataset_importances.append(dataset_importance_dict[f]) 
        else:
            dataset_importances.append(1)

    probs_accumulated = get_accumulated_probabilities(probs_list, dataset_importances)

    save_as_pickle(probs_accumulated, out_fname)

    return

def get_accumulated_probabilities(probs_list, dataset_importances = None):
    counter = 0

    # probs_accumulated = get_accumulated_probabilities
    for probs, weigth in zip(probs_list, dataset_importances):
        if type(dataset_importances) != type(None):
            probs *= weigth

        if counter == 0:
            probs_accumulated = probs
        else:
            if not set(list(probs.columns.values)).issubset(probs_accumulated.columns.values):
                # add the missing columns 
                for c in probs.columns.values:
                    if c not in probs_accumulated.columns.values: 
                        probs_accumulated[c] = np.zeros(len(probs_accumulated))
                
            probs_accumulated = probs_accumulated.add(probs, fill_value=0)
        counter += 1

    if type(dataset_importances) != type(None):
        if sum(np.array(dataset_importances)) <= 1:
            probs_accumulated *= len(dataset_importances) 

    probs_accumulated = probs_accumulated.div(counter)

    if INDEX_COL not in probs_accumulated:
        probs_accumulated[INDEX_COL] = probs_accumulated.index

    return probs_accumulated


def probabilities_from_path_to_classification(probabilities_fname): #, predictions_out_fname):
    '''
    Convert probabilities to predictions from a probabilities path.
    '''
    probabilities_df = load_from_pickle(probabilities_fname)
    predictions_df = probabilities_df_to_classification(probabilities_df)
    return predictions_df

def probabilities_df_to_classification(probabilities_df): 
    '''
    Convert probabilities to predictions from a probabilities path.
    '''
    if INDEX_COL in probabilities_df.columns.values:
        probabilities_df = probabilities_df.set_index(INDEX_COL)
    probabilities_df.columns = probabilities_df.columns.values.astype("int32")
    predictions_series = probabilities_df.idxmax(axis = 1)
    predictions_series.name = PREDICTION_COL
    predictions_df = predictions_series.to_frame()
    return predictions_df

'''
def prediction_main(selected_feature_cols = [],
                    window_sizes_samples = WINDOW_SIZES,
                    window_size_samples = WINDOW_SIZE,
                    window_step_size_samples = STEP_SIZE,
                    metrics = METRICS,
                    signal_cols_features = SIGNAL_COLS_FEATURES,
                    models_out_dir = MODELS_DIR,
                    model_type = MODEL_TYPE,
                    val_split = VAL_SPLIT,
                    fixed_window = FIXED_WINDOW,
                    train_val_normalized_dir = TRAIN_VAL_NORMALIZED_DIR,
                    predictions_out_dir = PREDICTIONS_DIR):

    
    if fixed_window:
        window_sizes_samples = [window_size_samples]

    if len(selected_feature_cols) == 0:
        selected_feature_cols = get_output_feature_cols(signal_cols_features, window_sizes_samples, metrics)

    # train_val_normalized_dir = os.path.join(train_val_normalized_dir, get_window_size_str(window_size_samples))
    val_dir = get_val_dir(train_val_normalized_dir, val_split)

    models_dir = get_models_predictions_evaluation_out_dir(models_out_dir, model_type, window_step_size_samples, len(selected_feature_cols), fixed_window = fixed_window, window_size_samples=window_size_samples)
    predictions_out_dir = get_models_predictions_evaluation_out_dir(predictions_out_dir, model_type, window_step_size_samples, len(selected_feature_cols), fixed_window = fixed_window, window_size_samples=window_size_samples)
    
    if not os.path.exists(models_dir): 
        print(f"Didn't find models in {models_dir}")
        return
    if not os.path.exists(predictions_out_dir):
        os.makedirs(predictions_out_dir)

    val_files   = glob.glob(val_dir + "/AOS*.csv")
    model_files = glob.glob(models_dir + "/*.pkl")
    model_files = [m for m in model_files if "label_en" not in m ]

    predicting_start_time = process_time()

    for val_file in val_files:

        predictions_out_fname = os.path.join(predictions_out_dir, os.path.basename(val_file).replace("_gt_features_val.csv", "_predictions.csv"))
        probabilities_out_fname = os.path.join(predictions_out_dir, os.path.basename(val_file).replace("_gt_features_val.csv", "_probabilities.csv"))

        if os.path.exists(predictions_out_fname):
            continue
        
        print(f"Making predictions for validation file {val_file}")
        X_val, Y_val = get_data_labels_from_file(val_file, selected_feature_cols, is_training = False)

        indices = get_indices_from_file(val_file)

        model_predictions_out_dir = os.path.join(predictions_out_dir, os.path.basename(val_file).replace("_gt_features_val.csv", "")) # for individual predictions
        if not os.path.exists(model_predictions_out_dir):
            os.makedirs(model_predictions_out_dir)
        
        for model_filename in model_files:
            model_probabilities_out_fname = os.path.join(model_predictions_out_dir, os.path.basename(model_filename).replace(".pkl", "_model_probabilities.csv"))
            if os.path.exists(model_probabilities_out_fname):
                continue
            
            with open(model_filename, 'rb') as model_f:

                model = pkl.load(model_f)

                Y_pred_prob = predict_probability(model, X_val)

                Y_pred_prob.index = indices

                if INDEX_COL not in Y_pred_prob.columns.values:
                    Y_pred_prob[INDEX_COL] = Y_pred_prob.index

                Y_pred_prob.to_csv(model_probabilities_out_fname, index = False)


        combine_classifications(model_predictions_out_dir, probabilities_out_fname)

        probabilities_to_classification(probabilities_out_fname, predictions_out_fname)
            
    predicting_stop_time = process_time()

    # Create output information
    # print_to_file(output_specs_file, (f"\nTraining for the following files took a total of {predicting_stop_time - predicting_start_time} seconds using a window step size of {window_step_size_samples}."))
    # for f in val_files:
    #     print_to_file(output_specs_file, f) 



    # for val_file in val_files: 
        
    #     X_val, Y_val = get_data_labels_from_file(val_file)
    #     indices = get_indices_from_file(val_file)

    #     # Predictions 
    #     prediction_start_time = process_time()
    #     Y_pred = predict(model, X_val)
    #     Y_pred_prob = predict_probability(model, X_val)
    #     prediction_stop_time = process_time() 
    #     print(f"Prediction ({len(X_val)}) samples took ", prediction_stop_time - prediction_start_time, " seconds.")

    #     # Evaluation
    #     ground_truth_vs_predictions(Y_train, Y_pred, y_pred_prob = Y_pred_prob, val_df = None, classes = None)

    #     predictions_output_dir = os.path.join(PREDICTIONS_DIR, os.path.basename(fname).replace("_gt.csv", ""))
    #     if not os.path.exists(predictions_output_dir):
    #         os.makedirs(predictions_output_dir)

    #     np.save(os.path.join(predictions_output_dir, "prediction_2.npy"), Y_pred)
    #     np.save(os.path.join(predictions_output_dir, "prediction_probability_2.npy"), Y_pred_prob)
    #     np.save(os.path.join(predictions_output_dir, "ground_truth_2.npy"), Y_val)
    #     val_df.to_csv(os.path.join(predictions_output_dir, "val_df_2.csv"))

    #     pickle.dump(model, open(os.path.join(predictions_output_dir, "svm.pkl"), 'wb'))

    #     print("")

if __name__ == "__main__":
    prediction_main()



## NN PART


def prediction_main_nn(activities_dir = ACTIVITIES_OUTPUT_DIR,
        models_dir = MODELS_DIR,
        predictions_dir = PREDICTIONS_DIR,
        model_type = NN_CONFIG["WINDOW_SIZE"], 
        signal_cols = SIGNAL_COLS,
        label_col = LABEL_COL,
        window_size_samples = NN_CONFIG["WINDOW_SIZE"],
        window_step_size = NN_CONFIG["STEP_SIZE"],
        epochs = NN_CONFIG["EPOCHS"],
        activity_dict = ACTIVITY_CODE_TYPE_DICT_CHRIS):

    n_classes = len(list(activity_dict.keys())) # all possible activities
    input_shape = (window_size_samples, len(signal_cols))

    # activities_dir = os.path.join(activities_dir, model_type)
    files = glob.glob(activities_dir + "/AOS*.csv")

    models_out_dir = get_nn_models_predictions_evaluation_out_dir(models_dir, model_type, window_size_samples, window_step_size, epochs)    
    predictions_out_dir = get_nn_models_predictions_evaluation_out_dir(predictions_dir, model_type, window_size_samples, window_step_size, epochs)
    if not os.path.exists(predictions_out_dir):
        os.makedirs(predictions_out_dir)
    
    for test_file in files:
        predictions_out_fname = get_nn_predictions_out_name(predictions_out_dir, test_file)

        if os.path.exists(predictions_out_fname):
            continue

        # Load the Model 
        model_out_fname = get_nn_model_out_name(models_out_dir, test_file,  False)
        if not os.path.exists(model_out_fname):
            print(f"Model does not exist: {model_out_fname}")
            continue
        print(f"Loading model from {model_out_fname} for predictions.")
        model = tf.keras.models.load_model(model_out_fname)
 
        test_dataset = tf.data.Dataset.from_generator(tf_data_generator, args= [[test_file], signal_cols, label_col, window_size_samples, 1, False], output_types = (tf.float32, tf.float32),
                                            output_shapes = ((None, window_size_samples, len(signal_cols)),(None, n_classes)))
        predictions = get_prediction_nn(model, test_dataset)

        # np.save(predictions_out_fname, predictions)

        predictions_out_df = combine_predictions_gt(test_file, predictions, window_size_samples)

        predictions_out_df.to_csv(predictions_out_fname, index = False)
'''