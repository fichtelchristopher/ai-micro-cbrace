from Configs.pipeline_config import *
from Configs.nn_pipeline_config import *
from Configs.shallow_pipeline_config import *
from classifier import *
from Models.neural_networks import *
from Models.neural_networks_configurations import *
from Processing.file_handling import get_model_basename_from_leave_out_test_file
from Processing.loading_saving import *
from optimization import *
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from Processing.file_handling import get_model_basename_from_leave_out_test_file
import pickle as pkl
from time import process_time
from classifier import *
from Processing.loading_saving import get_train_test_files
matplotlib.use("TkAgg")
from Analysis.transition_probability_analysis import get_transitions, filter_transition_df, get_transition_loc_indices
from Visualisation.visualisation import plot_default_gt

import matplotlib.pyplot as plt

classifier_type = "rf"

# In case you want to load from configs
# pipeline_config, classifier_config = get_configs_from_output_dir(models_dir)

pipeline_config = get_default_pipeline_config(classifier_type)
classifier_config = get_default_classifier_config(classifier_type)

# GET INPUT AND TEST FILES
db = "leipzig"
dataset = 1

# activities_dir = f"C:/Users/fichtel/Desktop/{db}/02-Activities-GT"
activities_dir = f"C:/Users/fichtel/Ottobock SE & Co. KGaA/Deep Learning - General/AI-Pipeline/{db}/02-Activities-GT"
features_dir = f"C:/Users/fichtel/Ottobock SE & Co. KGaA/Deep Learning - General/AI-Pipeline/{db}/04-Features-Normalized"
input_files = get_train_test_files(classifier_type, activities_output_dir = activities_dir, features_normalized_dir = features_dir, database = db)

# In case a pretrained should be loaded 
classifier_dir = "D:/AI-Pipeline/aos/DS1/05-Classifier/CNN_LSTM_SIMPLE/s1000_w32_e100_lr0.001_b64_C0/Training_1000"
classifier_type = "CNN_LSTM"
classifier_config = load_from_pickle(classifier_dir + "/classifier_config.pkl")
pipeline_config = load_from_pickle(classifier_dir + "/pipeline_config.pkl")
pipeline_config["STEP_SIZE_TEST"] = 1

for test_file in input_files: 
    if pipeline_config["TRAIN_PER_FILE"] == True:
        basename = get_model_basename_from_leave_out_test_file(test_file)
    else:
        basename = classifier_type + "_leave_out_" + os.path.basename(test_file).replace(".pkl", "")
    classifier = get_classifier(classifier_type, classifier_config, pipeline_config, basename)

    train_files = [f for f in input_files if f != test_file]
    history = classifier.train(train_files)
    
    for k, i in history.items():
        print(f"{k}:\t{i}")

    # # Predict and evaluate
    predictions, probabilities =  classifier.predict(test_file)
    acc_test, f1_test, loss_test = classifier.evaluate(test_file)
    print(f"test ccc:\t{acc_test}")
    print(f"test f1:\t{f1_test}")
    print(f"test loss:\t{loss_test}")
    
    # VISUALIZATION
    if classifier.classifier_type in NN_IMPLEMENTATIONS:
        # Visualisation
        classifier.visualize_prediction_vs_gt(test_file)
    else:
        # Get the ground truth as the train file only contains the features but not the signal data 
        test_file_gt =  test_file.replace("04-Features-Normalized", "02-Activities-GT")
        test_file_gt = test_file_gt.replace("_gt_features.pkl", "_gt_reduced.pkl")
        # # Visualisation
        classifier.visualize_prediction_vs_gt(test_file, test_file_gt)
    
    plt.show()
    plt.close()


    # _________________________________________________FROM HERE ON TRANSITIONS_________________________________________________________________________________
    # Transitions params
    transition_range = 100
    from_code = 100
    to_code = 0

    activity_code_type_dict = get_dir_activity_code_type_dict(Path(test_file).parent)

    # Load ground truth, feature df and get loc indices based on transition params
    data_df = load_from_pickle(test_file_gt)
    feature_df = load_from_pickle(test_file)
    transition_loc_indices = get_transition_loc_indices(feature_df, 64, transition_range, from_code = from_code, to_code = to_code, discard_swing_extension=True, activity_code_type_dict = activity_code_type_dict)

    for transition_loc_index in transition_loc_indices:
        transition_loc_indices_range = np.arange(transition_loc_index - transition_range, transition_loc_index + transition_range)

        transition_probabilities = probabilities.loc[transition_loc_indices_range]
        transition_predictions   = predictions.loc[transition_loc_indices_range]
        transition_data          = data_df.loc[transition_loc_indices_range]

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
        plt.sca(axs[0])
        plot_default_gt(transition_data, 
                        data_cols = ["JOINT_ANGLE", "JOINT_LOAD"], 
                        data_labels = ["JOINT ANGLE", "JOINT LOAD"],
                        data_scaling_factors = [1, 1/100],
                        _plot_indices = False)

        plt.axvline(x=transition_loc_index, color='black', linestyle='--')

        plt.sca(axs[1])
        plt.title(f"Transition from {activity_code_type_dict[from_code]} to {activity_code_type_dict[to_code]}")
        for activity, probability in transition_probabilities.items():
            activity_str = activity_code_type_dict[activity]
            plt.plot(probability, label = f"probability-{activity_str}")

        plt.legend()
        plt.grid(visible = True, axis = "y")
        plt.ylim((-0.05, 1.1))
        plt.axvline(x=transition_loc_index, color='black', linestyle='--')

        plt.tight_layout()
        plt.show()

        plt.close()
