''' 
Master thesis project of Christopher Fichtel started in Oct 2022
AI for microprocessor-controlled knee joints

File created 09 Nov 2022

Code Base for generating a reduced dataset. On the one hand dataset reduction means that 
phases with very little activity will be thrown away. Additionally, by providing the 
required parameters, one can rename or combine classes or also define classes to be
discarded for further processing. This can be done in the reduce_data method. For an 
example usage refer to the run_pipeline_preparation file where for the two databases 
(aos and leipzig) the code demonstrates the usage of the method. 
'''
import numpy as np
import pandas as pd

import glob
import os
import sys

from Processing.file_handling import get_leipig_files_from_dir, get_aos_files_from_dir

from Analysis.transitions_overview import * 
from Misc.utils import *
from Processing.loading_saving import *
from Processing.file_handling import * 
from AOS_SD_analysis.AOS_Rule_Set_Parser import * 
from Configs.pipeline_config import *
from Configs.shallow_pipeline_config import *

from Processing.loading_saving import *

from Misc.dataset_utils import *
from Misc.ai_utils import *

from Visualisation.visualisation import plot_default_gt

from Misc.feature_extraction import *
from Analysis.aos_data_gt_activity_analysis import * 

def get_new_activtiy_code_type_dict(input_activity_code_type_dict, discard_classes_code, combine_classes):
    '''
    :param discard_classes: list of [input_class_code: int, ...] --> these classes will have a mapping to class -100
    :param combine classes: [(input_class_code_1: int, input_class_code_2: int, ...), output_code: int , output_type: str]
    '''


    if (len(discard_classes_code) == 0) & (len(combine_classes) == 0):
        mapping = {}
        for activity_code in input_activity_code_type_dict.keys():
            mapping[activity_code] = activity_code

        return input_activity_code_type_dict, mapping

    mapping = {}
    output_activity_code_type_dict = {}
    output_activity_code_type_dict[DISCARD_CLASS_CODE] = DISCARD_CLASS_TYPE

    changed_class_codes_list = []

    for discard_class_code in discard_classes_code: 
        mapping[discard_class_code] = DISCARD_CLASS_CODE
        changed_class_codes_list.append(discard_class_code)

    for c in combine_classes:
        for input_class_code in c[0]:
            mapped_class_code = c[1]
            mapping[input_class_code] = mapped_class_code
            changed_class_codes_list.append(input_class_code)
        output_activity_code_type_dict[c[1]] = c[2]
    
    for code, type in input_activity_code_type_dict.items():
        if code not in changed_class_codes_list:
            output_activity_code_type_dict[code] = type
            mapping[code] = code

    output_activity_code_type_dict = {i: output_activity_code_type_dict[i] for i in sorted(output_activity_code_type_dict)}
    
    return output_activity_code_type_dict, mapping

def reduce_data(input_dir, output_dir, discard_classes = [], combine_classes = [], database = "aos"):
    '''
    Read all files from the input directory and create a "reduced" version for each of them, containing an ACTIVITY_IDX - column. See
    The create_reduced_df method. 

    (["200", "300"], "200", "level-walking")

    :param mode: either "aos" or "leipzig"
    '''

    input_activity_code_type_dict = get_dir_activity_code_type_dict(input_dir)

    output_activity_code_type_dict, mapping = get_new_activtiy_code_type_dict(input_activity_code_type_dict, discard_classes, combine_classes)

    # Save the output dictionary an mapping
    save_activity_code_type_dict(output_activity_code_type_dict, output_dir)
    save_activity_code_type_dict(mapping, output_dir, out_basename="mapping")

    output_file = os.path.join(output_dir, "data_reduction_overview.txt") 

    if database == "aos": 
        input_files = get_aos_files_from_dir(input_dir)
    else:
        input_files = get_leipig_files_from_dir(input_dir)

    # Create the reduced dataframes if desired
    for fname in input_files:
        fname_reduced = os.path.join(output_dir, os.path.basename(fname).replace("_gt", "_gt_reduced"))
    
        if not os.path.exists(fname_reduced):
            data_df = load_from_pickle(fname)
            data_df.set_index(INDEX_COL)
            # data_df = pd.read_csv(fname, index_col = INDEX_COL)
            # Optional visualisation of the input ground truth file
            # plot_default_gt(data_df)
            # plt.grid(visible = True, axis = "y")
            # plt.legend()
            # plt.title(os.path.basename(fname))
            # plt.show()
            # plt.clf()

            initial_length_samples = len(data_df)
            data_df_reduced = create_reduced_df(data_df, output_activity_code_type_dict, max_activity_duration_sec = MAX_ACTIVITY_DURATION_SEC, activity_transition_time_sec = ACTIVITY_TRANSITION_TIME_SEC, mapping = mapping) # this now has a meaningful "ACTIVITY_IDX" column, before all values are 0 in this column 
            save_as_pickle(data_df_reduced, fname_reduced)
            # data_df_reduced.to_csv(fname_reduced)
            print(f"Write reduced dataframe: {fname_reduced}")

            # Optional visualisation of the reduced ground truth file
            # plt.clf()
            # plot_default_gt(data_df_reduced)
            # plt.grid(visible = True, axis = "y")
            # plt.legend()
            # plt.title(os.path.basename(fname))
            # plt.show()

            out_length_samples = len(data_df_reduced)
            
            original_stdout = sys.stdout
            mode = "a" if os.path.exists(output_file) else "w"
            with open(output_file, mode) as f:
                sys.stdout = f # Change the standard output to the file we created.
                print(f"Input file: {fname}")
                print(f"Output file: {fname_reduced}")
                print(f"{np.round(100 * out_length_samples / initial_length_samples, 1)} % of initial data maintained after data reduction.")
                print(f"Max activity duration {MAX_ACTIVITY_DURATION_SEC},  Activity transition time {ACTIVITY_TRANSITION_TIME_SEC}.\n\n\n")
            sys.stdout = original_stdout
    
    dir_activity_analysis(output_dir, output_activity_code_type_dict, database = database)

    generate_transition_overview(output_dir, database = database)