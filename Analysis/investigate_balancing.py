from Configs.pipeline_config import *
from Configs.nn_pipeline_config import *
from Configs.shallow_pipeline_config import *
from classifier import *
from Models.neural_networks import *
from Models.neural_networks_configurations import *

from Processing.loading_saving import *
import matplotlib.pyplot as plt
from optimization import *
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from Processing.file_handling import get_model_basename_from_leave_out_test_file
from Misc.dataset_utils import get_labels_from_gt_loc_indices_files

from classifier import * 
from Processing.loading_saving import get_train_test_files

matplotlib.use("TkAgg")
#_____________________________________________________________________________
# gaus_fit_size = 512
# gaus_std_fac = 8
# std = gaus_fit_size / gaus_std_fac

# x = np.arange(-int(gaus_fit_size/2), int(gaus_fit_size/2))

# gaus_1 = norm.pdf(x, 0, std)
# gaus_1 /= np.max(gaus_1)
# gaus_2 = norm.pdf(x, 0, std)
# gaus_2 /= np.max(gaus_2)

# plt.plot(gaus_1, color = "red")
# # plt.plot(gaus_2, color = "blue")
# plt.show()
#______________________________________________________________________________  

classifier_type = "LSTM_SIMPLE"

# GET INPUT AND TEST FILES
db = "aos"
activities_dir = f"C:/Users/fichtel/Ottobock SE & Co. KGaA/Deep Learning - General/AI-Pipeline/{db}/02-Activities-GT"
input_files = get_train_test_files(classifier_type, activities_output_dir = activities_dir, features_normalized_dir = FEATURES_NORMALIZED_DIR, database = db)

activity_code_type_dict = get_dir_activity_code_type_dict(activities_dir)

# RUN CLASSIFICATIO N PIPELINE
window_size = 64
signal_cols = NN_SIGNALS
n_classes = len(activity_code_type_dict.keys())

step_size = 500
sampling_mode = "adaptive"

# Parameter for adaptive
undersample_on_label_distribution_shift = True
use_total_label_distribution = False

# Training dataset_range = (0, train_val_split), feature_file = False, use_total_label_distribution=True)
loc_indices_list, loc_indices_files_list = get_loc_indices_list_from_files(input_files, step_size = step_size, window_size=window_size, sampling_mode=sampling_mode, feature_file = False, dataset_range = (0, 0.7))
labels = get_labels_from_gt_loc_indices_files(input_files, loc_indices_files_list)
label_distribution_dict = get_label_distr_dict_from_activity_list(labels)
label_distribution_dict = apply_mapping_to_dict_keys(label_distribution_dict, activity_code_type_dict)

# Visualisation
sum_all_items = sum(label_distribution_dict.values())
visualize_dict_as_pie_chart(label_distribution_dict, labels = list(label_distribution_dict.keys()), title=f"Train dataset overview: {sum_all_items} total samples\n{sampling_mode}(step = {step_size})")
plt.show()

# Training dataset_range = (0, train_val_split), feature_file = False, use_total_label_distribution=True)
loc_indices_list, loc_indices_files_list = get_loc_indices_list_from_files(input_files, step_size = step_size, window_size=window_size, sampling_mode=sampling_mode, feature_file = False, dataset_range = (0.7, 1.0))
labels = get_labels_from_gt_loc_indices_files(input_files, loc_indices_files_list)
label_distribution_dict = get_label_distr_dict_from_activity_list(labels)
label_distribution_dict = apply_mapping_to_dict_keys(label_distribution_dict, activity_code_type_dict)

# Visualisation
sum_all_items = sum(label_distribution_dict.values())
visualize_dict_as_pie_chart(label_distribution_dict, labels = list(label_distribution_dict.keys()), title=f"Train dataset overview: {sum_all_items} total samples\n{sampling_mode}(step = {step_size})")
plt.show()