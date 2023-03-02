''' 
Master thesis project of Christopher Fichtel started in Oct 2022
AI in operating ortheses

File created 06 Oct 2022

Code base for detecting gait parameters (e.g. initial contacts)
'''

import numpy as np
import sys
from Misc.utils import *
from AOS_SD_analysis.AOS_Rule_Set_Libraries import *

from Configs.namespaces import *

def detect_swr(df, ruleset_type = "product"):
    '''
    Return the swing phase reversal indices.
    :param ic_indices: optional list of initial contacts. Will add a swr between every ic contact at the maximum of knee joint angle
    '''
    assert(set(["RI_RULID1"]).issubset(set(list(df.columns.values)))) # make sure all necessary signals are there

    df.reset_index(inplace = True)
    assert(df.index.values[0] == 0)

    ruleset = get_ruleset_key_code(ruleset_type)

    ri_rulid1_progression = compress_state_progression(df["RI_RULID1"].values)

    swext_progression_indices = np.where(np.array(ri_rulid1_progression[0], dtype = int) == ruleset["BASIS"][1]["SwExt"])[0]

    swr_indices = list(ri_rulid1_progression[1][swext_progression_indices])

    return swr_indices


def detect_swr_without_ruleset(df, ic_indices: list):
    '''
    :param df: the dataframe
    :param ic_indices: list of tuples [(ic_start_idx_1, ic_stop_idx_1), (ic_start_idx_2, ic_stop_idx_2),...,(...,...)]
    
    :return steps: a list of steps, described by (ic-start, )
    '''

    steps = []
    for ic_start_idx, ic_end_idx in ic_indices:

        data_df_masked = df.iloc[ic_start_idx + int(0.4 * (ic_end_idx - ic_start_idx)):ic_end_idx]

        duration = get_df_duration_in_seconds(data_df_masked)
        data_df_masked = data_df_masked["JOINT_ANGLE"]

        if duration > YIELDING_MAX_STEP_DURATION:
            continue
        
        if duration < LEVEL_MIN_STEP_DURATION:
            continue
        
        if data_df_masked.empty:
            continue

        joint_angle_max_idx = data_df_masked.idxmax()
    
        max_joint_angle = data_df_masked.max()
        if max_joint_angle > 10.0:
            steps.append((ic_start_idx, joint_angle_max_idx, ic_end_idx))

    return steps

def detect_ic(df, ruleset_type = "product"):
    '''
    
    :param df: preprocessed dataframe (see Processing/preprocessing.py file) containing the columns 
    :return: array with indices where initial contacts (supposedly) are
    '''
    assert(set(IC_DETECTION_NECESSARY_COLS).issubset(set(list(df.columns.values)))) # make sure all necessary signals are there

    df.reset_index(inplace = True)
    assert(df.index.values[0] == 0)

    ic_level_walking = ic_rule_level_walking(df, ruleset_type = ruleset_type)
    # x = df.index
    # plot_signals_from_df(df, data_cols = ["JOINT_ANGLE", "DDD_ACC_TOTAL", "RI_RULID1", "JOINT_ANGLE_VELOCITY", "JOINT_LOAD"], data_labels = ["JOINT_ANGLE", "ACC-TOTAL", "RI_RULID1", "JOINT_ANGLE_VELOCITY", "JOINT_LOAD"], data_scaling_factors = [1, 1, 1, 1 / 10, 1 / 100], x = x)
    # plt.vlines(x = ic_level_walking, ymin=-50, ymax = 50) 
    # plt.show()

    ic_yielding = ic_rule_yielding(df, ruleset_type = ruleset_type)
    # x = df.index
    # plot_signals_from_df(df, data_cols = ["JOINT_ANGLE", "DDD_ACC_TOTAL", "RI_RULID1", "JOINT_ANGLE_VELOCITY", "JOINT_LOAD"], data_labels = ["JOINT_ANGLE", "ACC-TOTAL", "RI_RULID1", "JOINT_ANGLE_VELOCITY", "JOINT_LOAD"], data_scaling_factors = [1, 1, 1, 1 / 10, 1 / 100], x = x)
    # plt.vlines(x = ic_yielding, ymin=-50, ymax = 50) 
    # plt.show()

    initial_contact_indices = comb_rule_indices(ic_yielding, ic_level_walking)

    # ic_rule_end_of_swing_ = ic_rule_end_of_swing(df, ruleset_type = ruleset_type) 

    # ic_rule_before_stance_ = ic_rule_before_stance(df, ruleset_type = ruleset_type)

    # initial_contact_indices = comb_rule_indices(ic_rule_end_of_swing_, ic_rule_before_stance_)

    # ic_rule_before_swing_no_stance_indices = ic_rule_before_swing_no_stance(df, ruleset_type = ruleset_type)

    # initial_contact_indices = comb_rule_indices(initial_contact_indices, ic_rule_before_swing_no_stance_indices)

    # ic_rule_before_yielding_indices =  ic_rule_before_yielding(df, ruleset_type = ruleset_type)

    # initial_contact_indices = comb_rule_indices(initial_contact_indices, ic_rule_before_yielding_indices)

    return initial_contact_indices


def detect_ic_acc_joint_angle_vel_based(df_masked, num_samples_around_acc_max = 10, max_joint_angle_at_ic = 20, ruleset_type = "product"):
    '''
    Based on a masked dataframe look for the maximum total acceleration value. If in the range of [idx-num_samples_around_acc_max:idx+num_samples_around_acc_max]
    there is a zero crossing of joint angle velocity, the index of the maximum value is the initial contact. If not, select the second highest
    acc val and so on. If the acc value is below 10, no ic contact is considered.
    '''
    ruleset = get_ruleset_key_code(ruleset_type)
    ruleset_basis_key_code = ruleset["BASIS"][1]

    ic_index = None
    acc_z_max = df_masked["DDD_ACC_TOTAL"].sort_values(ascending = False)
    acc_z_indices = acc_z_max.index
    for acc_val, idx in zip(acc_z_max, acc_z_indices): 
        if acc_val < 10: 
            break
        joint_angle_around_max = df_masked.loc[(df_masked.index > idx - num_samples_around_acc_max) & (df_masked.index < idx + num_samples_around_acc_max)]["JOINT_ANGLE_VELOCITY"].values
        zero_crossings = np.where(np.diff(np.sign(joint_angle_around_max)))[0]
        if len(zero_crossings) > 0:
            if df_masked.loc[df_masked.index == idx]["RI_RULID1"].values[0] not in [ruleset_basis_key_code["StanceExt"], ruleset_basis_key_code["SwFlex"]]:
                if df_masked.loc[df_masked.index == idx]["JOINT_ANGLE"].values[0] < max_joint_angle_at_ic:
                        ic_index = idx
                        break

    return ic_index

def ic_rule_level_walking(df, ruleset_type = "product", max_joint_angle_at_ic = 20):
    '''
    
    '''
    ic_indices = [-1]

    ruleset_key_code = get_ruleset_key_code(ruleset_type)
    ruleset_basis   = get_ruleset(ruleset_type)[ruleset_key_code["BASIS"][0]][1]

    activity_progression, start_indices = compress_state_progression(df["RI_RULID1"].values)

    step_rule1_progression_compressed_str   = [ruleset_basis[int(i)] for i in activity_progression]

    pattern_in_seq = find_pattern_in_sequence(step_rule1_progression_compressed_str, ["SwUnlock", "SwFlex", "SwExt"])

    before_sw_unlock = 100

    before_sw_ext_end = 50
    after_sw_ext_end = 30


    initial_contacts_after_swing = []
    for pattern_ind in pattern_in_seq:
        '''
        '''
        start_ind = start_indices[pattern_ind[0]]
        stop_ind = start_indices[pattern_ind[1]] - 1
        step_ic_indices = []
        # Get initial contact after swing
        df_masked_after_swext = df.iloc[stop_ind - before_sw_ext_end: stop_ind + after_sw_ext_end]
        ic_idx_post = detect_ic_acc_joint_angle_vel_based(df_masked_after_swext, ruleset_type=ruleset_type,  max_joint_angle_at_ic = max_joint_angle_at_ic)
        if type(ic_idx_post) != type(None):
            step_ic_indices.append(ic_idx_post)
            if ic_idx_post > ic_indices[-1]:
                initial_contacts_after_swing.append(ic_idx_post)
    
    initial_contacts_before_swing = []
    for pattern_ind in pattern_in_seq:
        '''
        '''
        start_ind = start_indices[pattern_ind[0]]
        stop_ind = start_indices[pattern_ind[1]] - 1

        step_ic_indices = []

        # Get initial contact before swing
        df_masked_before_swunlock = df.iloc[start_ind - before_sw_unlock : start_ind]
        ic_idx_prev = detect_ic_acc_joint_angle_vel_based(df_masked_before_swunlock, ruleset_type=ruleset_type,  max_joint_angle_at_ic = max_joint_angle_at_ic)
        if type(ic_idx_prev) != type(None):
            step_ic_indices.append(ic_idx_prev)
            if ic_idx_prev > ic_indices[-1]:
                initial_contacts_before_swing.append(ic_idx_prev)

    ic_indices = comb_rule_indices(initial_contacts_after_swing, initial_contacts_before_swing)
        # Plot step 100 before and after if desired
        # step_df = df.iloc[start_ind-100:stop_ind+100]
        # x = step_df.index
        # plot_signals_from_df(step_df, data_cols = ["JOINT_ANGLE", "DDD_ACC_TOTAL", "RI_RULID1", "JOINT_ANGLE_VELOCITY", "JOINT_LOAD"], data_labels = ["JOINT_ANGLE", "ACC-TOTAL", "RI_RULID1", "JOINT_ANGLE_VELOCITY", "JOINT_LOAD"], data_scaling_factors = [1, 1, 1, 1 / 10, 1 / 100], x = x)
        # plt.vlines(x = step_ic_indices, ymin=-10, ymax = 10) 
        # plt.show()
        # print("")
    
    ic_indices = ic_indices[1:]

    return ic_indices

def ic_rule_yielding(df, ruleset_type = "product", max_joint_angle_at_ic = 40):
    '''
    
    '''
    ic_indices = [-1]

    ruleset_key_code = get_ruleset_key_code(ruleset_type)
    ruleset_basis   = get_ruleset(ruleset_type)[ruleset_key_code["BASIS"][0]][1]

    activity_progression, start_indices = compress_state_progression(df["RI_RULID1"].values)

    step_rule1_progression_compressed_str   = [ruleset_basis[int(i)] for i in activity_progression]

    pattern_in_seq = find_pattern_in_sequence(step_rule1_progression_compressed_str, ["Yielding", "YieldingEnd", "SwExt"])

    before_yielding = 100

    before_sw_ext_end = 10
    after_sw_ext_end = 30

    for pattern_ind in pattern_in_seq:
        '''
        '''
        start_ind = start_indices[pattern_ind[0]]
        stop_ind = start_indices[pattern_ind[1]] - 1

        step_ic_indices = []

        # Get initial contact before swing
        df_masked_before_swunlock = df.iloc[start_ind - before_yielding : start_ind]
        ic_idx_prev = detect_ic_acc_joint_angle_vel_based(df_masked_before_swunlock, ruleset_type=ruleset_type, max_joint_angle_at_ic = max_joint_angle_at_ic)
        if type(ic_idx_prev) != type(None):
            step_ic_indices.append(ic_idx_prev)
            if ic_idx_prev > ic_indices[-1]:
                ic_indices.append(ic_idx_prev)

        # Get initial contact after swing
        df_masked_after_swext = df.iloc[stop_ind - before_sw_ext_end: stop_ind + after_sw_ext_end]
        ic_idx_post = detect_ic_acc_joint_angle_vel_based(df_masked_after_swext, ruleset_type=ruleset_type, max_joint_angle_at_ic = max_joint_angle_at_ic)
        if type(ic_idx_post) != type(None):
            step_ic_indices.append(ic_idx_post)
            if ic_idx_post > ic_indices[-1]:
                ic_indices.append(ic_idx_post)

    return ic_indices

    
def filter_ic_indices(df, initial_contact_indices, max_joint_angle = 15,ruleset_type = "product"):
    '''
    Filter out all inidices where the joint angle at the initial contact index is greater than the max joint angle

    :param df: input dataframe with "JOINT_ANGLE" property 
    :param initial_contact_indices: list of indices for initial contact
    :param max_joint_angle: the maximum joint angle at level walking   
    '''
    ruleset = get_ruleset_key_code(ruleset_type)
    
    filtered_ic_indices_angle = [i for i in initial_contact_indices if df.iloc[i]["JOINT_ANGLE"] < max_joint_angle]

    basis_memset = ruleset["BASIS"][1]
    filtered_ic_indices = []
    for i in filtered_ic_indices_angle:
        if df.iloc[i]["RI_RULID1"] not in [basis_memset["StanceExt"], basis_memset["SwFlex"], basis_memset["YieldingEnd"]]:
            filtered_ic_indices.append(i)

    return filtered_ic_indices

def ic_rule_end_of_swing(df, ruleset_type = "product"):
    '''
    Rule 1

    Initial contact is where the total acceleration is maximum after level walking swing phase
    range to look in: end of swing phase : end of swing phase + 20 samples (but before next stance phase) 20samples = 0.2sec
    '''  
    ruleset = get_ruleset_key_code(ruleset_type)

    level_walking_swing_indices = np.where(df["RI_RULID1"] == ruleset["BASIS"][1]["SwExt"])[0]

    level_walking_swing_start_indices, level_walking_swing_stop_indices = get_start_stop_indices(level_walking_swing_indices)

    initial_contact_indices = []

    num_samples_after_swing_stop = 30
    num_samples_before_swing_stop = 10

    stance_indices = np.where(df["RI_RULID1"] == ruleset["BASIS"][1]["StanceFlex"])[0]

    for swing_start_index, swing_stop_index in zip(level_walking_swing_start_indices, level_walking_swing_stop_indices):

        # Define range before stance flexion phase begins to look for acceleration
        start_range = swing_stop_index - num_samples_before_swing_stop
        
        try:
            ind_next_stance =  next(x for x in stance_indices if x > start_range)  
        except: 
            ind_next_stance = len(df)

        stop_range  = min(swing_stop_index + num_samples_after_swing_stop, ind_next_stance)

        df_masked = df.iloc[start_range:stop_range]

        if df_masked.empty:
            continue

        df_masked_swing_ext = df.iloc[swing_start_index:swing_stop_index]

        if df_masked[TOTAL_ACC_COL].max() * 2.5 < df_masked_swing_ext[TOTAL_ACC_COL].max():
            max_acc_index = df_masked_swing_ext[TOTAL_ACC_COL].idxmax()
        else:
            max_acc_index = df_masked[TOTAL_ACC_COL].idxmax()
        
        initial_contact_indices.append(max_acc_index)

    initial_contact_indices = filter_ic_indices(df, initial_contact_indices, ruleset_type = ruleset_type, max_joint_angle = 7.5)

    return initial_contact_indices


def ic_rule_before_stance(df, ruleset_type = "product"):
    '''
    Rule 2

    Initial contact is where acc signal is maximum in a certain time window (currently 300ms = 30samples) before stance flexion
    '''     
    ruleset = get_ruleset_key_code(ruleset_type)

    stance_phase_flexion_indices = np.where(df["RI_RULID1"] == ruleset["BASIS"][1]["StanceFlex"])[0]
    stance_flexion_start_indices, stance_flexion_stop_indices = get_start_stop_indices(stance_phase_flexion_indices)

    # < 30
    num_samples_before_stance_start_range = 30 # look for initial contact as maximum of total acceleration in range x samples before - until stance flexion
    num_samples_before_stance_stop_range = 0

    initial_contact_indices = []

    for (stance_flexion_start_index, stance_flexion_stop_index) in zip(stance_flexion_start_indices, stance_flexion_stop_indices):

        # Define range before stance flexion phase begins to look for acceleration
        start_range = stance_flexion_start_index - num_samples_before_stance_start_range
        stop_range  = stance_flexion_start_index - num_samples_before_stance_stop_range

        df_masked = df.iloc[start_range:stop_range]

        if df_masked.empty:
            continue

        max_acc_index = df_masked[TOTAL_ACC_COL].idxmax()

        initial_contact_indices.append(max_acc_index)

    initial_contact_indices = filter_ic_indices(df, initial_contact_indices, ruleset_type = ruleset_type, max_joint_angle = 7.5)
    
    return initial_contact_indices

def ic_rule_before_swing_no_stance(df, ruleset_type = "product"):
    '''
    Rule 3

    Initial contact is where the total acceleration is maximum in a time window [-1sec:-0.3sek] before level walking swing phase 
    condition: no stance phase in this interval
    ''' 
    ruleset = get_ruleset_key_code(ruleset_type)

    level_walking_swing_indices = np.where(df["RI_RULID1"] == ruleset["BASIS"][1]["SwUnlock"])[0]
    level_walking_swing_start_indices, level_walking_swing_stop_indices = get_start_stop_indices(level_walking_swing_indices)

    initial_contact_indices = []

    num_samples_before_swing_start_range_min = 70   # 0.5 seconds before the swing starts
    num_samples_before_swing_start_range_max = 10    # 0.01 seconds before the swing starts

    stance_indices_1 = np.where(df["RI_RULID1"] == ruleset["BASIS"][1]["StanceExt"])[0]
    stance_indices_2 = np.where(df["RI_RULID1"] == ruleset["BASIS"][1]["StanceFun"])[0]

    stance_indices = np.concatenate((stance_indices_1, stance_indices_2))
    stance_indices = []

    for swing_start_index in level_walking_swing_start_indices:

        # Define range before stance flexion phase begins to look for acceleration
        start_range = swing_start_index - num_samples_before_swing_start_range_min
        stop_range  = swing_start_index - num_samples_before_swing_start_range_max
        
        stance_in_range =  np.where(np.logical_and(start_range < np.array(stance_indices), np.array(stance_indices) < stop_range))[0]

        if stance_in_range.any():
            continue

        df_masked = df.iloc[start_range:stop_range]

        if df_masked.empty:
            continue

        max_acc_index = df_masked[TOTAL_ACC_COL].idxmax()

        initial_contact_indices.append(max_acc_index)
    
    initial_contact_indices = filter_ic_indices(df, initial_contact_indices,  ruleset_type = ruleset_type, max_joint_angle = 7.5)

    return initial_contact_indices

def ic_rule_before_yielding(df, ruleset_type = "product", max_joint_angle = 15):
    '''
    Rule 4

    If StanceFun is followed by a yielding step --> there is a IC before

    Example: 
    '''  
    initial_contact_indices = []

    ruleset = get_ruleset_key_code(ruleset_type)

    stance_fun_indices = np.where(df["RI_RULID1"] == ruleset["BASIS"][1]["StanceFun"])[0]
    if stance_fun_indices[-1] >= len(df) -1:
        stance_fun_indices = stance_fun_indices[:-1]

    stance_fun_start_indices, stance_fun_stop_indices = get_start_stop_indices(stance_fun_indices)
 
    for (stance_fun_start_index, stance_fun_stop_index) in zip(stance_fun_start_indices, stance_fun_stop_indices):
        
        # if ((stance_fun_start_index > 130300) & (stance_fun_start_index < 130600)):
        #     print("")
        if df["RI_RULID1"][stance_fun_stop_index + 1] == ruleset["BASIS"][1]["Yielding"]:

            df_masked = df.iloc[stance_fun_start_index-50:stance_fun_stop_index]

            if df_masked.empty:
                continue

            acc_z_knee_vel_zero = df_masked.loc[df_masked["JOINT_ANGLE_VELOCITY"] == 0]["DDD_ACC_Z"].sort_values(ascending = False)

            if acc_z_knee_vel_zero.empty:
                continue

            idx = acc_z_knee_vel_zero.idxmax()

            for x in acc_z_knee_vel_zero.iteritems():
                if df_masked["JOINT_ANGLE"][x[0]] < max_joint_angle:
                    initial_contact_indices.append(idx)
                    break

    return initial_contact_indices 


# def ic_rule_during_stance_fun(df, ruleset_type = "product"):
#     '''
#     Rule 5

#     During "StanceFun" look for the maximum knee angle. From this point look 50 seconds into the future and select the maximumg
#     of total acceleration 

#     Example: AOS170819_WK0201 typical range, index 156500 - 158500
#     '''  
#     initial_contact_indices = []

#     ruleset = get_ruleset_key_code(ruleset_type)

#     stance_fun_indices = np.where(df["RI_RULID1"] == ruleset["BASIS"][1]["StanceFun"])[0]
#     stance_fun_start_indices, stance_fun_stop_indices = get_start_stop_indices(stance_fun_indices)
 
#     num_samples_after_knee_max_angle = 50
#     num_samples_before_knee_max_angle = 100

#     initial_contact_indices = []

#     for (stance_fun_start_index, stance_fun_stop_index) in zip(stance_fun_start_indices, stance_fun_stop_indices):

#         df_masked_knee_angle_max = df.iloc[stance_fun_start_index-20:stance_fun_stop_index]
        
#         if df_masked_knee_angle_max.empty:
#             continue

#         max_angle = df_masked_knee_angle_max["JOINT_ANGLE"].max()
#         if (max_angle < 3):
#             continue

#         max_angle_index = df_masked_knee_angle_max["JOINT_ANGLE"].idxmax()

#         df_masked =  df.iloc[max_angle_index:max_angle_index + num_samples_after_knee_max_angle]
#         z_acc_max = df_masked["DDD_ACC_Z"].max()

#         knee_angle_at_z_acc_max = df_masked["JOINT_ANGLE"][df_masked["DDD_ACC_Z"].idxmax()]

#         if (z_acc_max > 12.5) or ((z_acc_max > 10) & (max_angle > 10)):
#             initial_contact_indices.append(df_masked["DDD_ACC_Z"].idxmax())

#         if (max_angle > 100):
#             df_masked =  df.iloc[max_angle_index-num_samples_before_knee_max_angle:max_angle_index]
#             z_acc_max = df_masked["DDD_ACC_Z"].max()
#             knee_angle_at_z_acc_max = df_masked["JOINT_ANGLE"][df_masked["DDD_ACC_Z"].idxmax()]

#             if knee_angle_at_z_acc_max < 0.3 * max_angle:
#                 initial_contact_indices.append(df_masked["DDD_ACC_Z"].idxmax())
    
#     initial_contact_indices = filter_ic_indices(df, initial_contact_indices, ruleset_type = ruleset_type)

#     return initial_contact_indices 

def comb_rule_indices(rule_1_indices, rule_2_indices, range = 50):
    '''
    Check for duplicate ic-indices. Returns a list of combined indices where rule 1 indices will be preferred over rule 2 indices
    when having "duplicates" in a time window of x samples (--> define x via range)

    :param rule_1_indices: initial contact indices determined by rule 1
    :param rule_2_indices: initial contact indices determined by rule 2
    :range: in samples (100Hz --> 100samples = 1sek), range in which to check if there are duplicates, if so select the index of the prefered rul e
    
    :return: list, combined indices
    '''
    rule_1 = list(np.full(len(rule_1_indices), fill_value=1))
    rule_2 = list(np.full(len(rule_2_indices), fill_value=2))

    rule_indices    = rule_1_indices + rule_2_indices
    rule            = rule_1 + rule_2

    rule_indices, rule = zip(*sorted(zip(rule_indices, rule)))

    # find "duplicate indices", i.e. difference between to initial contact detections is smaller than the selected range
    rule_indices_dif = np.array(rule_indices[1:]) - np.array(rule_indices[:-1])
    duplicate_indices = np.where(rule_indices_dif < range)[0]

    del_indices = []

    for duplicate_index in duplicate_indices:

        if (rule[duplicate_index] == 1):
            del_indices.append(duplicate_index + 1)
        else:
            del_indices.append(duplicate_index)

    rule_indices = np.delete(rule_indices, del_indices)
    return list(rule_indices)


"""
def rule_1_initial_contact_roland(df):
    '''
    Rule 1 

    Initial contact is where acc signal is maximum before stance-phase-flexion (600)
    condition: stance phase flexion phase is followed by "level-walking-swing-phase" or "level-walking-swing-phase-knee-max-X"
    '''

    stance_phase_flexion_indices = np.where(df[ACTIVITY_COL] == 600)[0]
    stance_flexion_start_indices, stance_flexion_stop_indices = get_start_stop_indices(stance_phase_flexion_indices)

    level_walking_swing_indices = np.where((df[ACTIVITY_COL].values >= 1000) & (df[ACTIVITY_COL].values < 2000) )[0]
    level_walking_swing_start_indices, level_walking_swing_stop_indices = get_start_stop_indices(level_walking_swing_indices)
    
    # < 100 
    num_samples_after_stance_stop = 100 # look for swing phase beginning
    # < 50
    num_samples_before_stance_start = 50 # look for initial contact as maximum of total acceleration in range x samples before stance flexion

    initial_contact_indices = []

    for (stance_flexion_start_index, stance_flexion_stop_index) in zip(stance_flexion_start_indices, stance_flexion_stop_indices):

        if any(x in level_walking_swing_start_indices for x in range(stance_flexion_stop_index, stance_flexion_stop_index + num_samples_after_stance_stop)):

            # Define range before stance flexion phase begins to look for acceleration
            start_range = stance_flexion_start_index - num_samples_before_stance_start
            stop_range  = stance_flexion_start_index

            df_masked = df.iloc[start_range:stop_range]

            max_acc_index = df_masked[TOTAL_ACC_COL].idxmax()

            initial_contact_indices.append(max_acc_index)
    
    return initial_contact_indices


def rule_2_initial_contact_roland(df):
    '''
    Rule 2

    Initial contact is where the joint load is maximum after level walking swing phase (code >= 1000 and code < 2000)
    range to look in: end of swing phase : end of swing phase + 100 samples (but before next stance phase)

    '''
    level_walking_swing_indices = np.where((df[ACTIVITY_COL].values >= 1000) & (df[ACTIVITY_COL].values < 2000) )[0]
    level_walking_swing_start_indices, level_walking_swing_stop_indices = get_start_stop_indices(level_walking_swing_indices)

    initial_contact_indices = []

    num_samples_after_swing_stop = 100

    stance_indices = np.where(df[ACTIVITY_COL].values == 600)[0]

    for swing_stop_index in level_walking_swing_stop_indices:

        # Define range before stance flexion phase begins to look for acceleration
        start_range = swing_stop_index
        
        try:
            ind_next_stance =  next(x for x in stance_indices if x > start_range)  
        except: 
            ind_next_stance = len(df)

        stop_range  = min(swing_stop_index + num_samples_after_swing_stop, ind_next_stance)

        df_masked = df.iloc[start_range:stop_range]

        max_joint_index = df_masked[TOTAL_ACC_COL].idxmax()

        initial_contact_indices.append(max_joint_index)
    
    return initial_contact_indices

def rule_3_initial_contact_roland(df):
    '''
    Rule 3

    Initial contact is where the total acceleration is maximum in a time window [-1sec:-0.3sek] before level walking swing phase (code >= 1000 and code < 2000)
    condition: no stance phase (code = 600) in this interval

    ''' 

    level_walking_swing_indices = np.where((df[ACTIVITY_COL].values >= 1000) & (df[ACTIVITY_COL].values < 2000) )[0]
    level_walking_swing_start_indices, level_walking_swing_stop_indices = get_start_stop_indices(level_walking_swing_indices)

    initial_contact_indices = []

    num_samples_before_swing_start_range_min = 100 # 1 second before the swing starts
    num_samples_before_swing_start_range_max = 30  # 0.3 seconds before the swing starts

    stance_indices = np.where(df[ACTIVITY_COL].values == 600)[0]

    for swing_start_index in level_walking_swing_start_indices:

        # Define range before stance flexion phase begins to look for acceleration
        start_range = swing_start_index - num_samples_before_swing_start_range_min
        stop_range  = swing_start_index - num_samples_before_swing_start_range_max
        
        stance_in_range =  np.where(np.logical_and(start_range < np.array(stance_indices), np.array(stance_indices) < stop_range))[0]

        if stance_in_range.any():
            continue

        df_masked = df.iloc[start_range:stop_range]

        max_acc_index = df_masked[TOTAL_ACC_COL].idxmax()

        initial_contact_indices.append(max_acc_index)


    return initial_contact_indices


"""
