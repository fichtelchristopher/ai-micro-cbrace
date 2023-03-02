''' 
Master thesis project of Christopher Fichtel (christopher.fichtel@ottobock.com) started in Oct 2022
AI in operating ortheses

File created 07 Oct 2022

Python Code Base adapted from the matlab code "AOS_Batch_RuleAnalysis" created by roland.auberger@ottobock.com
Code determines rules current "prothesis operation mode" (e.g. "walking", "stairs up", "ramp down" etc.)

Returned variables:
activity - interpretation of values
     0 ... no activity
   600 ... stance phase flexion
  1000 ... swing phase released without bending
  10X0 ... swing phase with maximum knee angle >X
  20X0 ... yielding ramp down with maximum knee angle >X
  3000 ... stumble
  4000 ... sit
  4100 ... intuitive stance
  4200 ... deliberate stance
  5000 ... training mode
  5100 ... freeze mode
  5200 ... MyMode
 -1000 ... new activity, not yet classified

 Matlab code explanation: 
    "K"     --> knee angle value
    "F"     --> hydraulic force
    "RTC"   --> real time clock


The following "dictionary" information can be found in the AOS_Batch_Analysis.mat file
Left: the namespaces used in the matlab code, right the corresponding names in csv/mat files
MATLAB code     CSV/MAT column names
col.ACC(1)      --> "DDD_ACC_X"
col.ACC(2)      --> "DDD_ACC_Y"
col.ACC(3)      --> "DDD_ACC_Z"
col.SID$        --> "RI_SETID$" with $ being an integer
col.RID$        --> "RI_RULID$" with $ being an integer
col.RTC         --> "RTC_HHMMSSZS" real time clock values
col.K           --> "JOINT_ANGLE"
col.KD          --> "JOINT_ANGLE_VELOCITY
col.FH          --> "JOINT_LOAD"
col.SFLEX       --> "SET_FLEX" Sollposition Flexionsdämpfung
col.SEXT        --> "SET_EXT" Sollposition Extensionsdämpfung
col.temp        --> "HYDRAULIC_TEMPERATURE"
col.aaccu       --> "ACCU_CAPACITY"
col.M_K         --> "KNEE_MOMENT"
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

# key = activity type (e.g. "level-walking-swing-phase")
# val = activitiy code (e.g.1000)
ACTIVITY_TYPE_CODE_DICT_ROLAND = {
    "no-activity": "0",
    "stance-phase-flexion": "600",
    "level-walking-swing-phase": "1000",
    "level-walking-swing-phase-knee-max-X": "10X0",
    "yielding": "2000",
    "yielding-knee-max-X": "20X0", 
    "stumbling": "3000",
    "sitting": "4000", 
    "intuitive-stance": "4100",
    "deliberate-stance": "4200",
    "training": "5000",
    "freeze": "5100",
    "my-mode": "5200",
    "new-activity": "-1000" 
}

def get_activity_progression_from_df(aos_data_df, ruleset_key_idx):
    '''
    For every point in time get the corresponding acitivity as described in the activity_dict at the beginning of this file.

    :param aos_data_df: pandas dataframe created with load_file()/load_mat_file()/load_csv_file() in the loading.py
            has to contain the columns for 
                - knee angle
                - force
                - accelartion
            in order to determine the acceleration
    :param ruleset: dictionary assigning the set id (int) a mode (str)
    :return: numpy array that has the same length as the dataframe, i.e. assigning every point in time an activity
    '''
    activity = np.zeros(len(aos_data_df))
    activity[0] = -1000 # initial activity is -1000 

    for i in tqdm(range(1, len(aos_data_df))):
        
        sid1 = int(aos_data_df["RI_SETID1"].iloc[i])
        sid2 = int(aos_data_df["RI_SETID2"].iloc[i])
        sid3 = int(aos_data_df["RI_SETID3"].iloc[i])

        rid1 = int(aos_data_df["RI_RULID1"].iloc[i])
        rid1_prev = int(aos_data_df["RI_RULID1"].iloc[i-1])
        rid2 = int(aos_data_df["RI_RULID2"].iloc[i])
        rid3 = int(aos_data_df["RI_RULID3"].iloc[i])
        rid3_prev = int(aos_data_df["RI_RULID3"].iloc[i-1])

        acc_var = aos_data_df["ACC_VAR"].iloc[i]  # variance has been computed before and added as column to dataframe

        sflex = aos_data_df["SET_FLEX"].iloc[i] # Sollposition Flexionsdämpfung
        sflex_prev = aos_data_df["SET_FLEX"].iloc[i-1]

        # level walking SWP released
        if (sid1 == ruleset_key_idx["BASIS"][0]) & (rid1 == ruleset_key_idx["BASIS"][1]["SwFlex"]): 
            if activity[i-1] < 1000:
                activity[i] = int(ACTIVITY_TYPE_CODE_DICT_ROLAND["level-walking-swing-phase"])

                if i > 50 & i <len(aos_data_df) - 50:

                    FH_min = np.min(aos_data_df["JOINT_LOAD"].iloc[ i-20:i+20] * cal_FH)
                    FH_max = np.max(aos_data_df["JOINT_LOAD"].iloc[ i-50:i+10] * cal_FH)
                    Mk_min = np.min(aos_data_df[KNEE_MOMENT_COL].iloc[i-20:i+20])
                    Mk_max = np.max(aos_data_df[KNEE_MOMENT_COL].iloc[i-50:i+10])
                    K_max  = np.max(aos_data_df["JOINT_ANGLE"].iloc[   i:i+50])

                    # round down to at least 10° 
                    act_code    = int(ACTIVITY_TYPE_CODE_DICT_ROLAND["level-walking-swing-phase"]) + 10 * floor(K_max / 10)
                    activity[i] = act_code
                    
            else: 
                activity[i] = activity[i - 1]
        # STPflexion standphasenflexing
        elif (sid1 == ruleset_key_idx["BASIS"][0]) & (rid1 == ruleset_key_idx["BASIS"][1]["StanceExt"]):
            activity[i] = int(ACTIVITY_TYPE_CODE_DICT_ROLAND["stance-phase-flexion"])

            if i > 50 & i <len(aos_data_df) - 50:

                FH_min = np.min(aos_data_df["JOINT_LOAD"].iloc[ i-20:i+30] * cal_FH)
                FH_max = np.max(aos_data_df["JOINT_LOAD"].iloc[ i-50:i+10] * cal_FH)
                Mk_min = np.min(aos_data_df[KNEE_MOMENT_COL].iloc[i-20:i+30])
                Mk_max = np.max(aos_data_df[KNEE_MOMENT_COL].iloc[i-50:i+10])
                K_max  = np.max(aos_data_df["JOINT_ANGLE"].iloc[i-30:i])

        # yielding ramp/stair descent
        elif (sid1 == ruleset_key_idx["BASIS"][0]) & (rid1 == ruleset_key_idx["BASIS"][1]["SwExt"]) &  (rid1_prev == ruleset_key_idx["BASIS"][1]["YieldingEnd"]):
            activity[i] = int(ACTIVITY_TYPE_CODE_DICT_ROLAND["yielding"])

            if i > 50 & i <len(aos_data_df) - 70:

                FH_min = np.min(aos_data_df["JOINT_LOAD"].iloc[ i-20:i+50] * cal_FH)
                FH_max = np.max(aos_data_df["JOINT_LOAD"].iloc[ i-10:i+70] * cal_FH)
                Mk_min = np.min(aos_data_df[KNEE_MOMENT_COL].iloc[i-20:i+50])
                Mk_max = np.max(aos_data_df[KNEE_MOMENT_COL].iloc[i-10:i+70])
                K_max  = np.max(aos_data_df["JOINT_ANGLE"].iloc[   i:i+15])

                # round down to at least 10°
                act_code    =  int(ACTIVITY_TYPE_CODE_DICT_ROLAND["yielding"]) + 10 * floor(K_max / 10)
                activity[i] = act_code

        # stumble SWP
        elif (sid1 == ruleset_key_idx["BASIS"][0]) & (rid1 == ruleset_key_idx["BASIS"][1]["Stumbled"]):
            activity[i] = int(ACTIVITY_TYPE_CODE_DICT_ROLAND["stumbling"])
            
            if i > 100 & i <len(aos_data_df) - 150:

                FH_min = np.min(aos_data_df["JOINT_LOAD"].iloc[ i-20 :i+50] * cal_FH)
                FH_max = np.max(aos_data_df["JOINT_LOAD"].iloc[ i-10 :i+70] * cal_FH)
                Mk_min = np.min(aos_data_df[KNEE_MOMENT_COL].iloc[i-20 :i+50])
                Mk_max = np.max(aos_data_df[KNEE_MOMENT_COL].iloc[i-10 :i+70])
                K_max  = np.max(aos_data_df["JOINT_ANGLE"].iloc[i-100:i+150])


        # sitting mode
        elif (sid3 == ruleset_key_idx["MODECHANGE"][1]) & (rid3 == ruleset_key_idx["MODECHANGE"][1]["Sit"]) & (rid3_prev == ruleset_key_idx["MODECHANGE"][1]["SitLoad"]) & (acc_var > 0.06):
            if (activity[i - 1] != int(ACTIVITY_TYPE_CODE_DICT_ROLAND["sitting"])) & a > 1:
                activity[i] = int(ACTIVITY_TYPE_CODE_DICT_ROLAND["sitting"])
            else:
                activity[i] = activity[i-1]
        # intuitive stance
        elif(sid2 == ruleset_key_idx["STANCEFUN"][0]) & (rid2 == ruleset_key_idx["STANCEFUN"][1]["AutoActive"]) & (sflex_prev == 100) & (acc_var > 0.06):
            if (activity[i - 1] != int(ACTIVITY_TYPE_CODE_DICT_ROLAND["intuitive-stance"])) & a > 1:
                activity[i] = int(ACTIVITY_TYPE_CODE_DICT_ROLAND["intuitive-stance"])
            else:
                activity[i] = activity[i-1]
        # deliberate_stance
        elif (sid2 == ruleset_key_idx["STANCEFUN"][0]) & (rid2 == ruleset_key_idx["STANCEFUN"][1]["DeliberateSF"]) & (sflex_prev == 100) & (acc_var > 0.06):
            if (activity[i - 1] != int(ACTIVITY_TYPE_CODE_DICT_ROLAND["deliberate-stance"])) & a > 1:
                activity[i] = int(ACTIVITY_TYPE_CODE_DICT_ROLAND["deliberate-stance"])
            else:
                activity[i] = activity[i-1]
        # training mode
        elif sid2 == ruleset_key_idx["TRAINING"][0]:
            activity[i] = int(ACTIVITY_TYPE_CODE_DICT_ROLAND["training"])
        # freeze mode
        elif sid2 == ruleset_key_idx["FREEZE"][0]:
            activity[i] = int(ACTIVITY_TYPE_CODE_DICT_ROLAND["freeze"])
        # my-mode
        elif sid2 == ruleset_key_idx["MYMODE"][0]: 
            activity[i] = int(ACTIVITY_TYPE_CODE_DICT_ROLAND["my-mode"])
        elif acc_var < 0.06:
            # no activity
            activity[i] = int(ACTIVITY_TYPE_CODE_DICT_ROLAND["no-activity"])
            a+=1
        elif activity[i-1] != -1000: 
            # new activity 
            activity[i] = int(ACTIVITY_TYPE_CODE_DICT_ROLAND["new-activity"])
            a+=1
        else:
            activity[i] = activity[i-1]
    
    return activity