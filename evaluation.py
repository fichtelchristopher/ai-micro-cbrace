''' 
Master thesis project of Christopher Fichtel started in Oct 2022
AI for microprocessor-controlled knee joints

File created 09 Nov 2022

Code Base for evaluating predictions 
'''

import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import pickle as pkl
from Misc.utils import *
from Processing.loading_saving import *
from Processing.preprocessing import *
from Processing.file_handling import * 
from Configs.pipeline_config import *
from Visualisation.visualisation import * 
from Misc.feature_extraction import get_output_feature_cols
from keras.losses import categorical_crossentropy
from Misc.dataset_utils import get_one_hot_encoder_from_labels
# from Misc.dataset_utils import convert_labels

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score
from Configs.shallow_pipeline_config import *


def ground_truth_vs_predictions(y_true, y_pred, y_pred_prob = None, val_df = None, classes = None):
    '''
    
    '''
    y_true_set = list(set(list(y_true)))
    y_pred_set = list(set(list(y_pred)))

    plt.plot(y_true, label = "y_true")
    plt.plot(y_pred, label = "y_pred")
    plt.plot(val_df["JOINT_ANGLE"], label = "JOINT_ANGLE")
    # plt.plot(val_df[TOTAL_ACC_COL] * 10, label = TOTAL_ACC_COL)

    if type(classes) != type(None): 
        y_pred_prob = pd.DataFrame(data = y_pred_prob, columns=model.classes_)
        plt.plot(100 * y_pred_prob[200], label = "standing_prob ")

    plt.legend()

    plt.show()

    print("")

def get_gt_pred_dfs_from_files(activities_file, predictions_file, step_size = 1):
    '''
    The dataframe can be set into relation to another via the idx column.
    
    :param activities_file: the activitiy file contains the "big" ground truth file for a patient. 
    :param predictions_file: predictions (normally) has been made on a smaller subset, predictions saved in the predicitons file. 

    :return: the loaded dataframes 
                data_df  
    '''
    try:
        data_df = pd.read_csv(activities_file, index_col=INDEX_COL)
    except:
        data_df = pd.read_csv(activities_file)

    predictions_df = pd.read_csv(predictions_file, index_col=INDEX_COL)

    data_gt_df = data_df[data_df.index.isin(predictions_df.index)]

    predictions_df[ACTIVITY_IDX_COL] = data_gt_df[ACTIVITY_IDX_COL] # add the activity index

    predictions_df.dropna(inplace=True)
    data_gt_df.dropna(inplace=True)

    return data_gt_df, predictions_df

def get_gt_pred_from_files(activites_file, predictions_file):
    '''
    :param activities_file: csv file, column name must contain LABEL_COL
    :param predictions_file: csv_file, corresponding predictions, column name must contain PREDICTION_COL 
    '''
    data_gt_df, predictions_df = get_gt_pred_dfs_from_files(activites_file, predictions_file)

    gt = data_gt_df[LABEL_COL]

    prediction = predictions_df[PREDICTION_COL]

    return gt, prediction

def get_evaluation_metrics(gt, prediction, activity_code_type_dict = ACTIVITY_CODE_TYPE_DICT_CHRIS, labels = None):
    '''
    :param gt: ground truth activity values as list or numpy array
    :param prediction: predicted activity values as list or numpy array

    Computes accuracy, f1 scores(multiclass macro) and a confusion matrix given an array of ground truth and corresponding predictions
    '''
    assert(len(gt) == len(prediction))

    available_labels = list(set(list(gt) + list(prediction)))
    display_labels = [activity_code_type_dict[x] for x in activity_code_type_dict.keys() if x in available_labels]

    accuracy = accuracy_score(gt, prediction)

    cm = confusion_matrix(gt, prediction) #, labels = list(ACTIVITY_TYPE_CODE_DICT_CHRIS.keys()) )

    cm_normalized = confusion_matrix(gt, prediction, normalize="true")
    
    # # For single label classification the f1 micro score is the same as accuracy
    # f1_score_micro = f1_score(gt, prediction, labels = labels, average='micro', zero_division='warn')

    # Macro score returns an objective measure of model performance also on imbalanced datasets
    f1_score_macro = f1_score(gt, prediction, labels = labels, average='macro', zero_division='warn')

    return accuracy, f1_score_macro, cm, cm_normalized, display_labels

def get_cross_entropy_loss(ground_truth, probability_df):
    '''
    '''
    if INDEX_COL in probability_df.columns.values: 
        probability_df = probability_df.set_index(INDEX_COL)

    probs_array = probability_df.to_numpy()

    one_hot_encoder = get_one_hot_encoder_from_labels(probability_df.columns.values)
    
    ground_truth = np.array(ground_truth, dtype = int)
    ground_truth_one_hot = one_hot_encoder.transform(ground_truth.reshape(-1, 1))

    ce_loss = categorical_crossentropy(ground_truth_one_hot, probs_array).numpy()

    ce_loss_mean = np.mean(ce_loss)

    return ce_loss_mean


def ground_truth_predictions_visualisations(activities_file, predictions_file, title = ""):
    '''
    
    '''
    data_gt_df_val, predictions_df = get_gt_pred_dfs_from_files(activities_file, predictions_file)

    for c, activity_idx in enumerate(list(set(list(predictions_df[ACTIVITY_IDX_COL].values)))):
        
        data_df_val_indexed = data_gt_df_val[data_gt_df_val[ACTIVITY_IDX_COL] == activity_idx]
        predictions_df_indexed = predictions_df[predictions_df[ACTIVITY_IDX_COL] == activity_idx]

        # data_indices = data_df_val_indexed.index.values
        # pred_indices = predictions_indexed.index.values

        signals = [LABEL_COL, "JOINT_ANGLE", TOTAL_ACC_COL, "KNEE_MOMENT", "RI_RULID1"]
        data_scaling_factors = [1, 1, 1, 1, 1] #10]

        activity_scaling_fac_plot = 1/10

        data_scaling_factors[0] = activity_scaling_fac_plot

        x = data_df_val_indexed.index.values / 100
        if c == 0:
            plot_signals_from_df(data_df_val_indexed, signals, x = x, data_scaling_factors=data_scaling_factors)
            plt.plot(x, predictions_df_indexed[PREDICTION_COL] * activity_scaling_fac_plot, label = PREDICTION_COL, color = "orange", linestyle = "--")
        else:
            plot_signals_from_df(data_df_val_indexed, signals, x = x, provide_labels=False, data_scaling_factors=data_scaling_factors)
            plt.plot(x, predictions_df_indexed[PREDICTION_COL] * activity_scaling_fac_plot, color = "orange", linestyle = "--")

    plt.legend()

    plt.title(title)

    plt.grid(visible = True, axis = "y")

    return

'''
def evaluation_main(selected_feature_cols = [],
                    window_sizes_samples = WINDOW_SIZES,
                    window_size_samples = WINDOW_SIZE,
                    window_step_size_samples = STEP_SIZE,
                    metrics = METRICS,
                    signal_cols_features = SIGNAL_COLS_FEATURES,
                    activities_dir  = ACTIVITIES_OUTPUT_DIR,
                    evaluation_out_dir = EVALUATION_DIR,
                    model_type = MODEL_TYPE,
                    fixed_window = FIXED_WINDOW,
                    predictions_out_dir = PREDICTIONS_DIR,
                    plot_evaluation = PLOT_EVALUATION):

    activity_code_type_dict = ACTIVITY_CODE_TYPE_DICT_CHRIS

    if fixed_window:
        window_sizes_samples = [window_size_samples]

    if len(selected_feature_cols) == 0:
        selected_feature_cols = get_output_feature_cols(signal_cols_features, window_sizes_samples, metrics)
    
    activity_gt_files = glob.glob(activities_dir + "/AOS*.csv")

    predictions_out_dir = get_models_predictions_evaluation_out_dir(predictions_out_dir, model_type, window_step_size_samples, len(selected_feature_cols), window_size_samples=window_size_samples, fixed_window = fixed_window)
    predictions_files = glob.glob(predictions_out_dir + "/AOS*predictions.csv")

    evaluation_out_dir = get_models_predictions_evaluation_out_dir(evaluation_out_dir, model_type, window_step_size_samples, len(selected_feature_cols), window_size_samples=window_size_samples, fixed_window = fixed_window)
    
    if not os.path.exists(evaluation_out_dir):
        os.makedirs(evaluation_out_dir)

    stat_out_file = os.path.join(evaluation_out_dir, "stat_overview.csv")
    stat_df = pd.DataFrame(columns=["file", "accuracy"])
    # Iterate over each activity file and evaluate
    for activity_gt_input_file in activity_gt_files:

        input_name = os.path.basename(activity_gt_input_file).replace("_gt.csv", "")
        input_name = input_name.replace("_gt_reduced.csv", "")

        for predictions_file in predictions_files:

            if input_name + "_predictions.csv" == os.path.basename(predictions_file):
                
                # Get evaluation results
                accuracy, cm, cm_normalized, display_labels = get_evaluation_metrics_from_files(activity_gt_input_file, predictions_file, activity_code_type_dict=activity_code_type_dict)

                stat_df = stat_df.append({"file": os.path.basename(activity_gt_input_file), "accuracy": accuracy}, ignore_index = True)

                # Visualize Confusion matrix and the normalized confusion matrix
                plot_confusion_matrix = True
                if plot_confusion_matrix:
                    subject_name = os.path.basename(predictions_file.replace("_predictions.csv", ""))
                    cm_title =  f"Confusion matrix for {subject_name}\naccuracy = {accuracy}"
                    out_save_fname = os.path.join(evaluation_out_dir, f"{subject_name}_cm.png")
                    visualize_confusion_matrix(cm, display_labels = display_labels, title = cm_title, out_save_fname = out_save_fname)
                    out_save_fname_normalized = out_save_fname.replace(".png", "_normalized.png")
                    visualize_confusion_matrix(cm_normalized, display_labels = display_labels, title = cm_title, out_save_fname = out_save_fname_normalized)

                # Visual evaluation via plots overlay
                plot_evaluation = False
                if plot_evaluation:
                    window_size_str = "" if fixed_window == False else f" window-size = {window_size_samples},"
                    title = f"Validation of {input_name},{window_size_str} step-size = {window_step_size_samples}, {len(selected_feature_cols)} features, accuracy = {accuracy}"
                    ground_truth_predictions_visualisations(activity_gt_input_file, predictions_file, title = title)
                    plt.show()
                    plt.clf()
   
    stat_df.to_csv(stat_out_file, index=False)
    print("")


if __name__ == "__main__":
    
    evaluation_main()
            

    s_dir = "C:/Users/fichtel/Ottobock SE & Co. KGaA/Deep Learning - General/Beispieldaten_AOS/AI-Pipeline/06-Predictions/AOS171117_WF0101L_200000-450000"

    gt_file = os.path.join(s_dir, "ground_truth_2.npy")
    
    pred_file = os.path.join(s_dir, "prediction_2.npy")

    pred_prob_file = os.path.join(s_dir, "prediction_probability_2.npy")

    val_df_file = os.path.join(s_dir, "val_df_2.csv")

    model_file = os.path.join(s_dir, "svm.pkl")

   

    y_true = np.load(gt_file)
    y_pred = np.load(pred_file)
    y_pred_prob = np.load(pred_prob_file)
    val_df = pd.read_csv(val_df_file)

    with open(model_file, 'rb') as f:
        model = pkl.load(f)
    
    classes = model.classes_

    ground_truth_vs_predictions(y_true, y_pred, y_pred_prob=y_pred_prob, val_df=val_df, classes=classes)


## NN EVALUATION
def evaluation_main_nn(evaluation_dir = EVALUATION_DIR,
        label_col = LABEL_COL):

    activities_dir = "C:/Users/FICHTEL/Ottobock SE & Co. KGaA/Deep Learning - General/Beispieldaten_AOS/AI-Pipeline/TYPICAL_RANGES/01-Activities-GT-Reduced"
    model_dir = "C:/Users/FICHTEL/Ottobock SE & Co. KGaA/Deep Learning - General/Beispieldaten_AOS/AI-Pipeline/TYPICAL_RANGES/05-Models-Reduced/CNN/s_50_w_100_e_50_lr_0.0001_batch_16_CONFIG0"

    classifier_type = os.path.basename(Path(model_dir).parent)
    with open(model_dir + "/pipeline_config.pkl", 'rb') as f:
        pipeline_configuration = pickle.load(f)
    with open(model_dir + "/classifier_config.pkl", 'rb') as f:
        classifier_configuration = pickle.load(f)

    p = "AOS180323_DM0104L_900000-1150000"
    model_path = model_dir + f"/{p}_leave_out.h5"
    test_file = activities_dir + f"/{p}_gt_reduced.csv"
    
    model = NeuralNetwork(classifier_type, classifier_configuration, pipeline_configuration, p)

    model.set_classifier_path(model_path)
    model.load_classifier_from_path()
    model.set_step_size_test(1)
    

    model.predict(test_file)

    test_f1_score = model.evaluate()

    model.visualize_prediction_vs_gt()
    '''