''' 
Master thesis project of Christopher Fichtel started in Oct 2022
AI in operating ortheses

File created 09 Nov 2022

Code base for steps utils
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from Misc.utils import *
from AOS_SD_analysis.AOS_Rule_Set_Libraries import *
from Configs.namespaces import *

# DONT CHANGE THE ORDER OF THESE! --> important for get_step_labels method
TRANSITIONS = {
        "yielding":  {"with_transition": [("Yielding", "YieldingEnd", "SwExt")], "without_transition": []}, #, ["Basis", "ToStanceFun", "StanceFun"]],
        "walking-w-flexion": {"with_transition": [("StanceFlex", "StanceExt"), ("SwFlex", "SwExt")], "without_transition": []},
        "walking-no-flexion": {"with_transition": [("SwFlex", "SwExt")], "without_transition": [(("StanceFlex", "StanceExt"))]}
    }


def get_steps_indices(data_df, ic_indices, swr_indices, level_min_step_duration = LEVEL_MIN_STEP_DURATION,  level_max_step_duration = LEVEL_MAX_STEP_DURATION, yielding_min_step_duration = YIELDING_MIN_STEP_DURATION, yielding_max_step_duration = YIELDING_MAX_STEP_DURATION, ruleset_type = "product"):
    '''
    A step is considered when the distance between two initial contacts lies in [min_step_duration, max_step_duration] and 
    if a swing phase reversal index lies between

    :param data_df: dataframe for which the initical contact and swing phase reversal indices were generated -->
                        only used for determining the number of samples for min/max step durations
    :param ic_indices: list of indices of initial contact
    :param swr_indices: list of indices where swing phase reversal takes place
    
    :param level_min_step_duraiton: minimum distance betweeen 2 initial contacts that might be considered a step
    :param level_max_step_duration: maximum distance between 2 initial contacts that might be considered a step

    
    :return: list of (start-idx, swr-idx, end-idx) 
    '''
    assert(SAMPLING_FREQ_COL in data_df.columns.values)
    assert("RI_RULID1" in data_df.columns.values)

    ruleset = get_ruleset_key_code(ruleset_type)

    steps_indices = []

    swr_indices = np.array(swr_indices)

    for ic_start_idx, ic_end_idx in zip(ic_indices[:-1], ic_indices[1:]):

        duration = get_df_duration_in_seconds(data_df, start_stop_index = (ic_start_idx, ic_end_idx))

        ri_rulid1 = data_df["RI_RULID1"].iloc[ic_start_idx:ic_end_idx]

        # only level walking has a swing phase with SwUnlock - SwFlex - SwExt - SwExtEnd 
        # sw_unlock_code  = ruleset["BASIS"][1]["SwUnlock"]
        sw_flex_code = ruleset["BASIS"][1]["SwFlex"]

        # if (sw_unlock_code in ri_rulid1.values) or (sw_flex_code in ri_rulid1.values):
        if (sw_flex_code in ri_rulid1.values):
            min_step_duration, max_step_duration = level_min_step_duration, level_max_step_duration
        else:
            min_step_duration, max_step_duration = yielding_min_step_duration, yielding_max_step_duration

        if min_step_duration <= duration <= max_step_duration:
            # check if initial contact index lies in between 
            swr_within = np.where((ic_start_idx < swr_indices) & (swr_indices < ic_end_idx))[0]

            if len(swr_within) == 1:

                steps_indices.append((ic_start_idx, swr_indices[swr_within[0]], ic_end_idx))

    return steps_indices


def get_steps_labels(data_df, steps_indices, ruleset_type = "product", activity_code_type_dict = ACTIVITY_CODE_TYPE_DICT_CHRIS, walking_class_ranges = WALKING_CLASSES_RANGES):
    '''
    
    '''
    activity_type_code_dict = get_reversed_dictionary(activity_code_type_dict)
    assert("RI_RULID1" in data_df.columns.values)

    labels = []

    for step_indices in steps_indices:
        start_idx = step_indices[0]
        stop_idx = step_indices[-1]

        step_rule1_progression = data_df["RI_RULID1"].iloc[start_idx:stop_idx]

        label_key = get_step_label_from_ruleprogression(step_rule1_progression, ruleset_type=ruleset_type)
        label = activity_type_code_dict[label_key]
        # Check maximum joint angle. If the angle is below a threshold --> assign "small step label"
        max_joint_angle = max(data_df["JOINT_ANGLE"].iloc[start_idx:stop_idx])
        
        label_added = False
        if label in WALKING_CLASSES_CODES_CHRIS:
            for idx, (range_min, range_max) in enumerate(walking_class_ranges):
                if (max_joint_angle >= range_min) & (max_joint_angle < range_max):
                    label += (idx + 1) *  10  
                    labels.append(label)
                    label_added = True
                    break
        
        if not label_added:
            labels.append(label)
        # if (max_joint_angle < SMALL_STEP_JOINT_ANGLE_TRESHOLD_LEVEL_WALKING) & (label in LEVEL_WALKING_CLASSES_CODES_CHRIS):
        #     label += 10
        #     assert(label in list(activity_code_type_dict.keys()))

        # if (max_joint_angle < SMALL_STEP_JOINT_ANGLE_TRESHOLD_YIELDING) & (label in YIELDING_CLASSES_CODES_CHRIS):
        #     label += 10
        #     assert(label in list(activity_code_type_dict.keys()))

        # labels.append(label)

    return labels


def get_step_label_from_ruleprogression(step_rule1_progression, ruleset_type = "product"):
    '''
    
    '''
    label = "other"

    ruleset_key_code = get_ruleset_key_code(ruleset_type)
    ruleset_basis   = get_ruleset(ruleset_type)[ruleset_key_code["BASIS"][0]][1]

    # Stumbling will be part of other class
    # If its desired to have a stumbling class --> define so in the "ACTIVITY_TYPE_CODE_DICT" and return "stumbling" instead of "other"
    if ruleset_key_code["BASIS"][1]["Stumbled"] in step_rule1_progression.values:
        return "other"  

    step_rule1_progression_compressed       = compress_state_progression(step_rule1_progression)[0]
    step_rule1_progression_compressed_str   = [ruleset_basis[int(i)] for i in step_rule1_progression_compressed]

    for key, item in TRANSITIONS.items():
        
        is_key = True

        with_transition_pattern = item["with_transition"]
        without_transition_pattern = item["without_transition"]

        for transition_pattern in with_transition_pattern:
            pattern_in_seq = find_pattern_in_sequence(step_rule1_progression_compressed_str, transition_pattern)
            if len(pattern_in_seq) == 0:
                is_key = False
                break

        for transition_pattern in without_transition_pattern:
            pattern_in_seq = find_pattern_in_sequence(step_rule1_progression_compressed_str, transition_pattern)
            if len(pattern_in_seq) > 0:
                is_key = False
                break
        
        if is_key:
            label = key

    return label

def get_standing_indices(data_df, acc_z_share_threshold = 0.8, max_joint_angle = 4, min_duration_sec = 3):
    '''
    Standing is considered when the share of z-acceleration in total-acceleration is above a certain threshold
    over a period of time while at the same time the joint angle is below a threshold.
    See the coordinate system of the C-brace for explanation of z as the axis of choice.  

    :param data_df: dataframe for which the standing indices should be found
    :param acc_z_share_threshold: the value above which the acc-z share has to be if considered standing
    :param max_joint_angle: the maximum joint angle if a phase should be considered standing

    :return: list of (start-idx, end-idx) 
    '''

    acc_z_share = data_df["DDD_ACC_Z"] / data_df[TOTAL_ACC_COL]

    over_treshold_mask = (acc_z_share > acc_z_share_threshold) & (data_df["JOINT_ANGLE"] < max_joint_angle)
    over_treshold_mask = over_treshold_mask.values
    over_treshold_mask = np.insert(over_treshold_mask, 0, -1) # to consider first as well

    sequence_start_indices = np.where(over_treshold_mask[1:] != over_treshold_mask[:-1])[0]
    sequence_start_indices += 1

    standing_start_stop_indices = []

    for start_idx, stop_idx in zip(sequence_start_indices[:-1], sequence_start_indices[1:]):

        if over_treshold_mask[start_idx] == 1:
            # we are in the "standing activity"
            time_passed_sec = np.sum(1 / data_df[SAMPLING_FREQ_COL].iloc[start_idx:stop_idx]) # in seconds

            if time_passed_sec > min_duration_sec: 
                standing_start_stop_indices.append((start_idx, stop_idx))

    return standing_start_stop_indices
