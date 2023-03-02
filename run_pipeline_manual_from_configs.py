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
from Processing.file_handling import get_model_name_from_leave_out_test_file

import matplotlib.pyplot as plt

# set the DATASET value in Configs.pipeline_config!

if DATASET == 1:
    configs_dict = {
                    "CNN_LSTM_SIMPLE": "D:/AI-Pipeline/leipzig/DS1/05-Classifier/CNN_LSTM_SIMPLE/s10_w32_e30_lr0.001_b64_C0",
                   "CNN_SIMPLE": "D:/AI-Pipeline/leipzig/DS1/05-Classifier/CNN_SIMPLE/s10_w64_e90_lr0.001_b128_C2",
                   "LSTM_SIMPLE": "D:/AI-Pipeline/leipzig/DS1/05-Classifier/LSTM_SIMPLE/s10_w64_e30_lr0.0001_b64_C0",
                   "xgboost": "D:/AI-Pipeline/leipzig/DS1/05-Classifier/xgboost/s_10_w_16_32_64_mutual_info_numf50_C19",
                   "lda": "D:/AI-Pipeline/leipzig/DS1/05-Classifier/lda/s_10_w_16_32_64_pca_numf75_C8",
                   "svm": "D:/AI-Pipeline/leipzig/DS1/05-Classifier/svm/s_10_w_16_32_64_pca_numf50_C9",
                   "rf": "D:/AI-Pipeline/leipzig/DS1/05-Classifier/rf/s_10_w_16_32_64_pca_numf75_C21"
    }
if DATASET in  [2, 3, 4, 5]:
    configs_dict = {
                    "CNN_LSTM_SIMPLE": "D:/AI-Pipeline/leipzig/DS2/05-Classifier/CNN_LSTM_SIMPLE/s10_w128_e30_lr0.001_b128_C2",
                   "CNN_SIMPLE": "D:/AI-Pipeline/leipzig/DS2/05-Classifier/CNN_SIMPLE/s10_w16_e60_lr0.001_b128_C0",
                   "LSTM_SIMPLE": "D:/AI-Pipeline/leipzig/DS2/05-Classifier/LSTM_SIMPLE/s10_w64_e60_lr0.001_b64_C0",
                   "xgboost": "D:/AI-Pipeline/leipzig/DS2/05-Classifier/xgboost/s_10_w_16_32_64_mutual_info_numf75_C44",
                   "lda": "D:/AI-Pipeline/leipzig/DS2/05-Classifier/lda/s_10_w_16_32_64_mutual_info_numf100_C1",
                   "svm": "D:/AI-Pipeline/leipzig/DS2/05-Classifier/svm/s_10_w_16_32_64_mutual_info_numf100_C4",
                   "rf": "D:/AI-Pipeline/leipzig/DS2/05-Classifier/rf/s_10_w_16_32_64_mutual_info_numf50_C3"
    }

def main():

    loso = True

    for classifier_type, output_dir in configs_dict.items():

        input_files = get_train_test_files(classifier_type)

        pipeline_config_f = output_dir + f"/pipeline_config.pkl"
        classifier_config_f = output_dir + f"/classifier_config.pkl"

        if loso:
            train_files_list = []
            basenames = []
            for i in input_files: 
                train_files_list.append([x for x in input_files if i != x])
                basename = os.path.basename(i).replace(".pkl", "")
                basenames.append(classifier_type + f"_leave_out_{basename}")

        else:
            train_files_list = [input_files]
            basename = classifier_type + f"_{len(input_files)}_train_files"
            basenames = [basename]

        for train_files, basename in zip(train_files_list, basenames):

            try:
                pipeline_config = load_from_pickle(pipeline_config_f)
                classifier_config = load_from_pickle(classifier_config_f)
            except:
                pipeline_config_f = pipeline_config_f.replace("D:/AI-Pipeline", "C:/Users/fichtel/Ottobock SE & Co. KGaA/Deep Learning - General/AI-Pipeline")
                classifier_config_f = classifier_config_f.replace("D:/AI-Pipeline", "C:/Users/fichtel/Ottobock SE & Co. KGaA/Deep Learning - General/AI-Pipeline")
                pipeline_config = load_from_pickle(pipeline_config_f)
                classifier_config = load_from_pickle(classifier_config_f)

            if classifier_type in SHALLOW_IMPLEMENTATIONS:
                pipeline_config["TRAIN_PER_FILE"] = True
            else:
                pipeline_config["EPOCHS"] = 100

            for step_size in [250, 500, 1000]:

                pipeline_config["STEP_SIZE"] = step_size

                # for sampling_mode in ["adaptive", "fixed"]:
                sampling_mode = "fixed"

                pipeline_config["SAMPLING_MODE"] = sampling_mode

                try:
                    classifier = get_classifier(classifier_type, classifier_config, pipeline_config, basename)
                    history = classifier.train(train_files)

                    if classifier_type in SHALLOW_IMPLEMENTATIONS:
                        output_classifier_fname = classifier.classifier_output_dir + f"/{classifier.classifier_name}.pkl"
                        save_as_pickle(classifier, output_classifier_fname)
                        print(f"Saved classifier in {output_classifier_fname}.")
                except: 
                    print("")

if __name__ == "__main__": 
    main()