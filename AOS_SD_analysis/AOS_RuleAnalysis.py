''' 
Master thesis project of Christopher Fichtel (christopher.fichtel@ottobock.com) started in Oct 2022
AI in operating ortheses

File created 20 Oct 2022

Python Code Base for generating an activity stream based on set and rule ids. 

Returned variables:
activity - interpretation of values
    0   ... "other" /unknown
    1   ... "sitting"
    2   ... "standing"
    3   ... "walking-no-flexion"
    4   ... "walking-w-flexion"
    5   ... "yielding"
    6   ... "stumbling"
'''
from math import floor
from Misc.utils import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from Configs.namespaces import *
from Configs.constants import cal_FH
import sys
from time import process_time


# For each activity define a typical step based on the rule sequence
# typical step refers to a full gait cycle when performing the respective activity longer than 1 step
# The "No_" prefix means that every other step is accepted
STEP_RULE_SEQUENCE_PATTERNS_STUDY = {
    "walking-no-flexion": {"BASIS": [["SwExtEnd", "SwUnlock", "SwFlex", "SwExt", "No_StanceFlex"],
                                    ["SwExtEnd", "SwUnlock", "Circumduction", "SwFlex", "SwExt", "No_StanceFlex"],
                                    ["SwUnlock", "SwFlex", "SwExt","SwExtEnd", "No_StanceFlex"],
                                    ["SwUnlock", "Circumduction", "SwFlex", "SwExt", "SwExtEnd", "No_StanceFlex"]]},
                                    

    "walking-w-flexion": {"BASIS": [["SwExtEnd", "StanceFlex", "StanceExt", "SwUnlock", "SwFlex", "SwExt"],
                                    ["SwExtEnd", "StanceFlex", "StanceExt", "SwUnlock", "Circumduction", "SwFlex", "SwExt"],
                                    ["SwUnlock", "SwFlex", "SwExt", "SwExtEnd", "StanceFlex", "StanceExt"],
                                    ["SwUnlock", "Circumduction", "SwFlex", "SwExt", "SwExtEnd", "StanceFlex", "StanceExt"]]},

    "walking-w-flexion-incomplete": {"BASIS": [["SwUnlock", "SwFlex", "SwExt", "SwExtEnd", "StanceFlex", "No_StanceExt"],
                                               ["SwUnlock", "Circumduction", "SwFlex", "SwExt", "SwExtEnd", "StanceFlex", "No_StanceExt"]]}, 

    "yielding": {"BASIS": [["SwExtEnd", "StanceFlex", "Yielding", "YieldingEnd", "SwExt"],
                            ["Yielding", "YieldingEnd", "SwExt", "SwExtEnd", "StanceFlex"]]},

    "sitting": {"SIT": [["Sit"]]},

    "standing": {"BASIS": [["ToBasis", "ToStanceFun", "StanceFun", "Unloaded", "StanceFun"], 
                           ["ToBasis", "Basis", "ToStanceFun", "StanceFun", "Unloaded", "StanceFun"],
                           ["ToBasis", "ToStanceFun", "StanceFun"],
                           ["ToBasis", "Basis", "ToStanceFun", "StanceFun"],
                           ["StanceFun", "Unloaded", "StanceFun"]]},

    "stumbling": {"BASIS": [["Stumbled"]]}
}


unique_sequence_transitions_study = {
    "walking-no-flexion": {"BASIS": [["SwExtEnd", "SwUnlock"],
                                    ["SwExtEnd", "ToBasis"]]},

    "walking-w-flexion":  {"BASIS": [["StanceFlex", "StanceExt"],
                                    ["StanceExt", "SwUnlock"],
                                    ["StanceExt", "ToBasis"]]},

    "yielding": {"BASIS":   [["Yielding", "YieldingEnd"],
                            ["YieldingEnd", "SwExt"],
                            ["StanceFlex", "Yielding"],
                            ["StanceFun", "Yielding"]]},
    "sitting": {"SIT":[["Sit"]]}, 
    "standing": {"BASIS": [ ["StanceFun", "Unloaded"],
                            ["Unloaded", "StanceFun"],
                            ["ToStanceFun", "StanceFun"],
                            ["StanceFun", "ToBasis"]]},
    "stumbling": {"BASIS": [["Stumbled", "SwExt"],
                        ["SwFlex", "Stumbled"],
                        ["SwExt", "Stumbled"]]}
}


# def get_activity_progression_from_df_chris(aos_data_df, ruleset, ruleset_key_idx, ruleset_type = "study", rule_progession_out_fname = ""):
#     '''
#     For every point in time get the corresponding acitivity as described in the activity_dict at the beginning of this file.

#     :param aos_data_df: pandas dataframe created with load_file()/load_mat_file()/load_csv_file() in the loading.py
#             has to contain the columns for 
#                 - knee angle
#                 - force
#                 - accelartion
#             in order to determine the acceleration
#     :param ruleset: dictionary assigning the set id (int) a mode (str)
#     :param ruleset_key_idx: dictionary assigning the mode (str) a set id (int)
#     ruleset_type: "study" or "product"
#     :return: numpy array that has the same length as the dataframe, i.e. assigning every point in time an activity
#     '''
#     if ruleset_type == "study":
#         activities_sequence_patterns = STEP_RULE_SEQUENCE_PATTERNS_STUDY 
#     elif ruleset_type == "product":
#         activities_sequence_patterns = FULL_STEP_RULE_SEQUENCE_PATTERNS_PRODUCT # TODO to be implemented
#     else:
#         print(f"Invalid ruleset type {ruleset_type}")
#         sys.exit()
    
#     activity = np.zeros(len(aos_data_df))   # other activity has index 0 
#     rules_progression = []

#     sid1_ = aos_data_df["RI_SETID1"].values.astype(int)
#     rid1_ = aos_data_df["RI_RULID1"].values.astype(int)

#     if os.path.exists(rule_progession_out_fname):
#         rules_progression = np.load(rule_progession_out_fname)
#     else:
#         for i in tqdm(range(1, len(aos_data_df))):

#             sid1 = sid1_[i]
#             rid1 = rid1_[i]

#             # BASIS rulset is saved in set id 1 --> this contains standing, yielding, level walking and stumbling; only sitting in other ruleset
#             if sid1 == ruleset_key_idx["BASIS"][0]: 
#                  rule = ruleset[sid1][1][rid1]
#             # SIT rulset is saved in set id 1
#             elif sid1 == ruleset_key_idx["SIT"][0]:
#                 rule = ruleset[sid1][1][rid1]
#             else: 
#                 rule = "other"

#             rules_progression.append(rule)
        
#         rules_progression = [rules_progression[0]] + rules_progression # as we started iterating at index 1

#         print("Finished computing rule progression.")

#         try:
#             np.save(rule_progession_out_fname, rules_progression)
#             print(f"Saved rule progression {rule_progession_out_fname}")
#         except:
#             print(f"Could not save {rule_progession_out_fname}. This is not a program crash but a loss of an interim result, so you should consider checking the filename again if saving is required.")

#     rule_sequence, rule_sequence_indices = compress_state_progression(rules_progression)

#     # activity_step_statistics = {}

#     for activity_name, activity_patterns in activities_sequence_patterns.items():
#         for ruleset_name, patterns in activity_patterns.items():
#             for pattern in patterns:
#                 if activity_name == "walking-w-flexion-incomplete":
#                     print("")
#                 start_stop_indices = find_pattern_in_sequence(rule_sequence, pattern)

#                 for start_idx, stop_idx in start_stop_indices:
#                     pattern_start_idx = rule_sequence_indices[start_idx]
#                     pattern_stop_idx = rule_sequence_indices[stop_idx]

#                     activity[pattern_start_idx:pattern_stop_idx] = ACTIVITY_TYPE_CODE_DICT_CHRIS[activity_name]

#                     # Get the transitions 
#                     if activity_name in ["walking-no-flexion", "walking-w-flexion", "walking-w-flexion-incomplete", "yielding"]:
#                         start_stop_indices_transition = find_transition_pattern_in_sequence(rule_sequence, pattern, start_idx, stop_idx)
#                         for start_idx_transition, stop_idx_transition in start_stop_indices_transition:
#                             transition_start_idx = rule_sequence_indices[start_idx_transition]
#                             transition_stop_idx = rule_sequence_indices[stop_idx_transition]

#                             if (transition_stop_idx-transition_start_idx) < MAX_TRANSITION_DURATION_SAMPLES:

#                                 activity[transition_start_idx:transition_stop_idx] = ACTIVITY_TYPE_CODE_DICT_CHRIS[activity_name + "-transition"]

#     for activity_name, transition_patterns in unique_sequence_transitions_study.items():
#         for ruleset_name, patterns in transition_patterns.items():
#             for pattern in patterns:
#                 start_stop_indices = find_pattern_in_sequence(rule_sequence, pattern)

#                 for start_idx, stop_idx in start_stop_indices:
#                     pattern_start_idx = rule_sequence_indices[start_idx]
#                     pattern_stop_idx = rule_sequence_indices[stop_idx]

#                     current_activity = activity[pattern_start_idx:pattern_stop_idx]
#                     if all(v == 0 for v in current_activity):
#                         activity[pattern_start_idx:pattern_stop_idx] = ACTIVITY_TYPE_CODE_DICT_CHRIS[activity_name]

#     print(f"Generated activities")
    
#     return activity, rules_progression


def find_transition_pattern_in_sequence(rule_sequence, pattern, activity_start_idx, activity_stop_idx):
    '''

    :param activity_start_idx: look before this index for transition
    :param activity_stop_idx: look after this index for transition
    '''
    transition_start_stop_indices = []

    # Check if pattern continues after stop idx
    counter = 0
    for i in range(activity_stop_idx, np.amin([activity_stop_idx + len(pattern), len(rule_sequence)])):
        if rule_sequence[i] == pattern[counter]:
            counter += 1
        else:
            break
    if counter > 0: 
        transition_start_stop_indices.append((activity_stop_idx, activity_stop_idx + counter))

    # Check if pattern continues (reversely) before start idx
    counter = 1
    for i in reversed(range(np.amax([0, activity_start_idx - len(pattern) - 2]), activity_start_idx)):
        if counter <= len(pattern):
            if rule_sequence[i] == pattern[-counter]:
                counter += 1
            elif rule_sequence[i] in ["Basis", "ToBasis"]:
                counter += 1
            else: 
                break
        elif rule_sequence[i] in ["Basis", "ToBasis"]:
            counter += 1
        else:
            break

    if counter > 1: 
        transition_start_stop_indices.append((activity_start_idx - counter + 1, activity_start_idx))


    return transition_start_stop_indices