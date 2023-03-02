''' 
Master thesis project of Christopher Fichtel started in Oct 2022
AI for microprocessor-controlled knee joints

File created 06 Oct 2022

Code Base for creating the ground truth labels for the determined classes

0 level walking
1 sitting
2 level walking w. stance phase flexion
3 level walking without stance phase flexion
4 yielding, i.e. ramp and stairs down
'''
import glob
from itertools import product
import os
import sys

from Gait.gait_params_detection import detect_ic
from Analysis.transitions_overview import * 

from Misc.utils import *
from Processing.file_handling import get_subject_name, get_subject_df_out_fname
from Analysis.aos_data_gt_activity_analysis import *
from Processing.loading_saving import *
from AOS_SD_analysis.AOS_Batch_RuleAnalysis_Roland import *
from AOS_SD_analysis.AOS_Subject import * 
from Processing.preprocessing import *
from Processing.loading_saving import load_from_pickle, save_as_pickle

from AOS_SD_analysis.AOS_Rule_Set_Parser import * 
from Configs.pipeline_config import *
from Visualisation.namespaces_visualisation import * 

from time import process_time

from Configs.pipeline_config import ACTIVITIES_OUTPUT_DIR
from Configs.namespaces import *

from Analysis.activity_analysis import *

def check_output_dir(code_type_dict, activities_full_output_dir):    
    '''
    Check if already a code type dict has been generated in this directory. 
    '''
    try:
        existing_code_type_dict = get_dir_activity_code_type_dict(activities_full_output_dir)
    except: 
        existing_code_type_dict = code_type_dict

    if existing_code_type_dict != code_type_dict:
        print("Recheck activity code type dict you are suppling and the already existing one in the activities_full_output_dir. You might consider chosing a different output directory.")
        sys.exit()

def generate_steps_overview(aos_subject: AOS_Subject):
    '''
    '''
    # Generate steps overview
    step_labels = aos_subject._steps_labels 
    activity_code_type_dict = aos_subject.activity_code_type_dict
    subject_df_out_fname = aos_subject.subject_df_out_fname
    subject_name = aos_subject._name
    
    label_dict = get_label_distr_dict(activity_code_type_dict, step_labels)
    codes_not_walking = []
    for code in label_dict.keys():
        if "walking" not in activity_code_type_dict[code]:
            if "yielding" not in activity_code_type_dict[code]:
                codes_not_walking.append(code)
    for code in codes_not_walking:
        del(label_dict[code])        

    duration = np.round( aos_subject._aos_data_df["TIME"].values[-1] / 3600, 2)

    visualize_dict_as_pie_chart(label_dict, output_file= subject_df_out_fname.replace(".pkl", "_step_overview.png"), labels = [activity_code_type_dict[k] for k in label_dict.keys()], title=f"Number of steps for {subject_name}\n{duration}~hours")
    plt.clf()

def ground_truth_generation_aos_main(raw_input_data_dir = DATA_DIR_EXAMPLE_DATA, activities_full_output_dir = ACTIVITIES_FULL_OUTPUT_DIR, subject_typical_ranges = SUBJECT_TYPICAL_RANGES, use_typical_ranges = USE_TYPICAL_RANGES, activity_code_type_dict = ACTIVITY_CODE_TYPE_DICT_CHRIS):

    if DATABASE != "aos":
        print("Wrong database")
        sys.exit()
    
    # generate_transition_overview(activities_full_output_dir, database = "aos") # this can be called if the data has already been generated

    check_output_dir(activity_code_type_dict, activities_full_output_dir)

    save_activity_code_type_dict(activity_code_type_dict, activities_full_output_dir)

    files = glob.glob(raw_input_data_dir + "/AOS*.mat")

    dir_activity_analysis(activities_full_output_dir, activity_code_type_dict=activity_code_type_dict)

    # Create a ground truth file for every file
    for f in files: #[files[1]]:
        subject_name  = get_subject_name(f)

        # Whether to use the typical ranges
        if use_typical_ranges: 
            range = subject_typical_ranges[subject_name]
        else:
            range = None

        # Load & Preprocess the data. Add time column, knee moment, delete "unneccessary" columns in order to save memory
        aos_data_df = load_data_file(f, range = range)  # supports mat, csv, txt file
        aos_data_df = preprocess_df(aos_data_df, knee_lever = KNEE_LEVER)
        
        # Define some names for generating subject/saving results
        ruleset_type            = get_ruleset_type_from_df(aos_data_df) # will return either "product" or "study"
        subject_df_out_fname    = get_subject_df_out_fname(activities_full_output_dir, f, range)

        # if os.path.exists(subject_df_out_fname):
        #     continue

        # Create an AOS subject instance.
        print(f"Getting activtiy for {subject_name}.")
        aos_subject = AOS_Subject(aos_data_df, subject_df_out_fname, ruleset_type = ruleset_type, name = subject_name) # this generates the ground truth file in subject_df_out_fname

        # Generate a step overview
        generate_steps_overview(aos_subject)

        aos_data_df = aos_subject._aos_data_df  # this now has an ACTIVITY_COL column
        ic_indices = aos_subject._ic_indices    # the initial contact indices
        swr_indices = aos_subject._swr_indices  # indices for swing phase reversal
        
        _plot_data = False
        if _plot_data: 
            matplotlib.use("TkAgg")
            # Plot some signals, ic indices and swing phase reversal indices

            activity = np.array(aos_data_df[LABEL_COL].values)
            # activity[np.where(activity == 500)] = 350

            activity[np.where(activity == 600)] = 0
            # activity[np.where(activity == 500)] = 400
            aos_data_df[LABEL_COL] = activity

            plot_signals_from_df(aos_data_df, data_cols = ["JOINT_ANGLE", "DDD_ACC_TOTAL", "RI_RULID1", LABEL_COL, "JOINT_ANGLE_VELOCITY"], data_labels = ["ANGLE", "ACC-TOTAL", "RI_RULID1", LABEL_COL, "JOINT_ANGLE_VELOCITY"], data_scaling_factors = [1 / 2, 1, 1, 1 / 10, 1 / 100])
           
            _plot_indices = True
            if _plot_indices:
                plot_indices(ic_indices, color = SIGNAL_COLOR_DICT[IC_COL], alpha = 1.0, ymax = 20, ymin = 0, label = IC_COL)
                plot_indices(swr_indices, color = SIGNAL_COLOR_DICT[SWR_COL], alpha = 1.0, ymax = 20, ymin = 0, label = SWR_COL)
            
            # Set plotting parameters
            plt.title(os.path.basename(f).replace(".mat", ""))
            plt.legend()
            plt.show()

            plt.clf()
            matplotlib.use("Agg")


    # Analyse files in this directory
    dir_activity_analysis(activities_full_output_dir, activity_code_type_dict=activity_code_type_dict)
    generate_transition_overview(activities_full_output_dir, database = "aos")

if __name__ == "__main__":
    
    ground_truth_generation_aos_main()
    
'''
# saves existing set-rule id combinations in set_rule_out_fname --> quick overview over existing states
set_rule_out_fname          = get_set_rule_output_fname(RULE_ID_OUTPUT_DIR, f)
set_rule_id_combinations    = get_set_rule_combinations(aos_data_df, output_file = set_rule_out_fname) 

activity_progression_out_fname          = get_activity_progression_output_fname(ACTIVITIES_INTERMEDIATE_RESULTS_DIR, f, range)
subject_activity_statistics_out_fname   = get_subject_activity_statistic_output_fname(ACTIVITIES_INTERMEDIATE_RESULTS_DIR, f, range)
rule_progression_out_fname              = get_subject_rule_progression_output_fname(ACTIVITIES_INTERMEDIATE_RESULTS_DIR, f, range)

level_walking_activities = aos_subject._get_activities_via_type("level-walking-swing-phase")
for activity in level_walking_activities:
    signal_values = activity._get_possible_signal_values()
    # activity.plot_signal("JOINT_ANGLE_VELOCITY")
    activity.plot_signals(["JOINT_ANGLE_VELOCITY"])
'''