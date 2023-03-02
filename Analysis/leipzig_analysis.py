''' 
Master thesis project of Christopher Fichtel started in Oct 2022
AI in operating ortheses

File created 06 Oct 2022

Script for analysing leipzig files
'''

from Misc.utils import *
from Processing.loading_saving import *
from AOS_SD_analysis.AOS_Subject import * 
from Processing.preprocessing import *
from Processing.file_handling import * 
from AOS_SD_analysis.AOS_Rule_Set_Parser import * 
from Configs.pipeline_config import *
from Visualisation.visualisation import *
from Visualisation.namespaces_visualisation import *


if __name__ == "__main__":
    
    data_files_leipzig_dict = get_data_files_leipzig_dict(DATA_DIR_LEIPZIG)

    for subject_key, subject_files in data_files_leipzig_dict.items():

        for f in subject_files:

            name = f.split("Leipzig2018Messungen")[-1]

            aos_data_df = load_data_file(f) # supports mat, csv, txt file

            # Get the labels dict containing the step indices ranges (label corresponds to the step type e.g. "RA", "LW", ...)
            label_files = get_leipzig_label_files(f)
            labels_indices_dict = get_labels_indices_dict(label_files)

            # Preprocess the data. Add time column, knee moment, delete "unneccessary" columns in order to save memory
            aos_data_df = preprocess_df(aos_data_df, knee_lever = KNEE_LEVER)
            print(aos_data_df.columns.values)

            plot_data_w_steps(aos_data_df, labels_indices_dict, data_cols = [TOTAL_ACC_COL, "JOINT_LOAD"],
                                                                data_colors = ["orange", "turquoise"],
                                                                data_labels = ["acc total", "(joint load) / 100"],
                                                                data_scaling_factors= [1, (1/100)],
                                                                title = name)


