''' 
Master thesis project of Christopher Fichtel (christopher.fichtel@ottobock.com) started in Oct 2022
AI in operating ortheses

File created 11 Oct 2022

Python Code Base for a subject, i.e. the data for one data file.  
'''

from tracemalloc import start
# from AOS_SD_analysis.AOS_Activity import AOS_Activity
from Misc.utils import *
from Processing.loading_saving import load_from_pickle, save_as_pickle
from Gait.gait_params_detection import * 
from Gait.steps import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from time import process_time
from tqdm import tqdm

from Configs.pipeline_config import * 

# from AOS_SD_analysis.AOS_Batch_RuleAnalysis_Roland import *
from AOS_SD_analysis.AOS_Rule_Set_Libraries import *
from AOS_SD_analysis.AOS_RuleAnalysis import * 

class AOS_Subject:
    '''
    Represents subject for the data of one file.
    '''
    def __init__(self, aos_data_df, subject_df_out_fname, ruleset_type = "product", name = None, activity_code_type_dict = ACTIVITY_CODE_TYPE_DICT_CHRIS):
        '''
        :param aos_data_df: the dataframe corresponding to one file, loaded via the load_data_file(filepath)
        :param activity_progression_out_fname: filepath where to save the numpy array representing activity indices (see method aos_generate_activity_progression)
        :param ruleset_type: either "study" or "product" (see RULE_SET_LIBRARIES in namespaces.py)
        :param activity_gen: "roland" (python implementation of his matlab code) or "chris" 
        '''
        assert(ruleset_type in RULE_SET_LIBRARY_TYPES)
        assert(subject_df_out_fname.endswith(".pkl"))

        self.activity_code_type_dict = activity_code_type_dict

        # Define subject name/alias
        self._name = name
        if name is None:
            self._name = os.path.basename(subject_df_out_fname)

        # Define output paths
        self.subject_df_out_fname = subject_df_out_fname

        # Define used ruleset
        self.ruleset            = get_ruleset(ruleset_type)            # int-code to rule-string dictionary
        self.ruleset_key_idx    = get_ruleset_key_code(ruleset_type)   # string (e.g. "BASIS") as dict key, code id in the rule set (e.g. 4 for study and 6 for product ruleset) as dict item
        self.ruleset_type       = ruleset_type

        # Load (if existing) or generate activity, ic and swr indices &
        # add to input dataframe as additional datastreams with columns ACTIVITY_COL, IC_COL, SWR_COL
        # IC and SWR columns will be zero except where IC/SWR is detect (--> 1)
        if os.path.exists(self.subject_df_out_fname):
            print(f"Loading from {self.subject_df_out_fname}")
            self.load_df() # loads into self._activity_df
            self._ic_indices = np.where(self._aos_data_df[IC_COL] == 1)[0]
            self._swr_indices = np.where(self._aos_data_df[SWR_COL] == 1)[0]
            self.load_steps_labels()
        else:
            # Initiate subject dataframe
            self._aos_data_df = aos_data_df[AOS_SUBJECT_COLS_PREV]

            # Get indices of the initial contacts and swing phase reversal 
            self._ic_indices = detect_ic(self._aos_data_df, ruleset_type = self.ruleset_type)
            self._swr_indices = detect_swr(self._aos_data_df, ruleset_type = self.ruleset_type)
            # ic_indices = [(start, stop) for start, stop in zip(self._ic_indices[-1], self._ic_indices[1:])]
            # steps = detect_swr_without_ruleset(self._aos_data_df, ic_indices)
            # self._swr_indices = [x[1] for x in steps]
            
            # get list of step (start, swr and end idx) and list of step (label)
            self.load_steps_indices()
            self.load_steps_labels()
            # self._standing_indices = get_standing_indices(self._aos_data_df)

            self.generate_streams()
            self._aos_data_df[IC_COL]         = self._ic_stream
            self._aos_data_df[SWR_COL]        = self._swr_stream
            self._aos_data_df[LABEL_COL]   = self._step_based_activity_stream
            # self._aos_data_df[ACTIVITY_COL_CLOSED ] = self._step_based_activity_stream_closed

            self._aos_data_df = self._aos_data_df[AOS_SUBJECT_COLS]# + [ACTIVITY_COL_CLOSED ]] 
            self.save_df()
            print(f"Saved to {self.subject_df_out_fname}")

    def load_steps_indices(self):
        '''
        '''
        self._steps_indices = get_steps_indices(self._aos_data_df, self._ic_indices, self._swr_indices, ruleset_type = self.ruleset_type)   # a list of step tuple (start, swr, end idx)

    def load_steps_labels(self):
        '''
        '''
        self.load_steps_indices()
        self._steps_labels = get_steps_labels(self._aos_data_df, self._steps_indices, ruleset_type = self.ruleset_type, activity_code_type_dict = self.activity_code_type_dict)

    def generate_streams(self):
        self.generate_ic_stream()
        self.generate_swr_stream()
        self.genereate_step_based_activity_stream()

    def aos_generate_acc_var(self):
        '''
        Add the acceleration variance as a column to the ddf
        '''
        # Compute the variations in batch ACCvar
        acc_var_batch_size = 1000
        acc_var = get_acc_batch_var(self._aos_dastta_df, acc_var_batch_size)

        # Add "ACC_VAR" columns
        self._aos_data_df["ACC_VAR"] = acc_var

    def genereate_step_based_activity_stream(self, activity_code_type_dict = ACTIVITY_CODE_TYPE_DICT_CHRIS, walking_classes_codes = WALKING_CLASSES_CODES_CHRIS):
        '''
        Based on list of step indices (self._steps_indices) and step labels (self._steps_labels) create 
        an activity stream i.e. 1 label per sample

        :param walking_classes_codes: list of walking codes (int) --> step label between two initial contacts only considered a step 
                                        if one of the walking classes types is active within this region
        '''
        activity_type_code_dict = get_reversed_dictionary(activity_code_type_dict)

        # check if swing extension is in activity code type dict
        swing_extension_code = 0 # default class 
        for code, type in activity_code_type_dict.items():
            if type == SWING_EXTENSION_CLASSNAME:
                swing_extension_code = code


        self._step_based_activity_stream = np.full(len(self._aos_data_df), fill_value=activity_type_code_dict["other"])

        # Add sitting --> where sampling frequency is 10 
        self._step_based_activity_stream[self._aos_data_df[SAMPLING_FREQ_COL] < SITTING_SAMPLING_FREQ_THRESHOLD] = activity_type_code_dict["sitting"]

        # Add standing --> define conditions 
        # for standing_indices in self._standing_indices:
        #     start_idx = standing_indices[0]
        #     end_idx = standing_indices[1]
        #     self._step_based_activity_stream[start_idx:end_idx] = activity_type_code_dict["standing"]

        # Add walking steps (level walking with/without stance phase flexion and yielding)
        for step_indices, step_label in tqdm(zip(self._steps_indices, self._steps_labels)):
            start_idx = step_indices[0]
            swr_idx = step_indices[1]
            end_idx = step_indices[2]

            self._step_based_activity_stream[start_idx:swr_idx] = step_label
            
            # Swing Extension only after a walking step
            if step_label in walking_classes_codes:
                self._step_based_activity_stream[swr_idx:end_idx] = swing_extension_code

        #  Add stumbling
        self._step_based_activity_stream[self._aos_data_df["RI_RULID1"] == get_ruleset_key_code(self.ruleset_type)["BASIS"][1]["Stumbled"]]= activity_type_code_dict["stumbling"]

        self._step_based_activity_stream = close_activity_gaps(self._step_based_activity_stream, max_gap = 5, max_gap_sitting=25)
    
    def generate_ic_stream(self):
        '''
        Stream of zeros with 1 at places where initial contact (IC) is detected.
        '''
        self._ic_stream = np.zeros(len(self._aos_data_df))
        self._ic_stream[self._ic_indices] = 1

    def generate_swr_stream(self):
        '''
        Stream of zeros with 1 at places where swing phase reversal (SWR) is detected.
        '''
        self._swr_stream = np.zeros(len(self._aos_data_df))
        self._swr_stream[self._swr_indices] = 1
      
    def save_df(self):
        '''
        Save the aos_data_df dataframe to the output_fname file (csv)
        '''
        # self._aos_data_df.to_csv(self.subject_df_out_fname, index = False)
        save_as_pickle(self._aos_data_df, self.subject_df_out_fname)
        return
    
    def load_df(self):
        '''
        Loads the aos_data_df dataframe from the output_fname file (csv)
        '''
        # self._aos_data_df = pd.read_csv(self.subject_df_out_fname)
        self._aos_data_df = load_from_pickle(self.subject_df_out_fname)
        return

    def _get_activity_progression(self):
        return self._activity_progression

    def _get_activities(self):
        return self._activities

    def _get_activities_via_code(self, activity_code):
        '''
        Select all activities via the activity code 
        '''
        act_type = get_reversed_dictionary()[activity_code]
        return self._get_activities_via_type(act_type)
        
    def _get_activities_via_type(self, activity_type): 
        '''
        Select all activities via the activity code 
        :return: list of activities (i.e. instances of AOS_Activity)
        '''
        indices = self._activities_indices[activity_type]
        return np.array(self._activities)[indices]

    def aos_generate_subject_activity_statistics_old(self, filename_txt_out = None):
        #TODO calc more activity statistics ? 
        ''' 
        Calculate some activity statistics 

        :param filename_txt_out: if provided --> save statistics to file
                                if None --> print statistics to console
        '''
        if os.path.exists(filename_txt_out): 
            os.remove(filename_txt_out)

        with open(filename_txt_out, "a") as f:
            for act_code in sorted(list(set(self._activities_codes))):
                print(f"{act_code}\t:{self._activities_codes.count(act_code)}", file = f)

    def aos_generate_subject_activity_statistics(self, filename_txt_out = None):
        '''
        Calculate subject specific activity statistics. 
        
        Includes total time per activity.
        '''
        if os.path.exists(filename_txt_out): 
            os.remove(filename_txt_out)

        activities = self._aos_data_df[LABEL_COL].values.astype(int)
        activities_set = sorted(list(set(list(activities))))

        self.total_duration = 0
        self._activities_durations = {}

        for act_code in activities_set:
            activity_duration = self._aos_data_df.loc[self._aos_data_df[LABEL_COL] == act_code, SAMPLING_FREQ_COL]
            activity_duration = 1 / activity_duration
            activity_duration = round(np.sum(activity_duration), 2) # in seconds 
            self.total_duration+=activity_duration
            self._activities_durations[act_code] = activity_duration

        with open(filename_txt_out, "a") as f:
            for act_code in activities_set:  
                activity_duration = self._activities_durations[act_code]
                activity = get_activity_from_int_code_chris(act_code)
                print(f"{activity} ({act_code}): {activity_duration} seconds", file = f)

            print(f"\nTotal duration: {self.total_duration} seconds", file = f)  

    """
    self._rule_progression_out_fname = rule_progression_out_fname

    def aos_generate_activity_progression(self):
        '''
        For each point in time assign an activity. Will be saved in self.activities_arr. 
        Either load from file (default, if file exists), otherwise create from df
        '''

        print("Generating activity.")
        t1_start = process_time() 
        if self._activity_gen_method == "roland":
            self._activity_progression = get_activity_progression_from_df(self._aos_data_df, self.ruleset_key_idx)
        elif self._activity_gen_method == "chris":
            self._activity_progression, self._rules_progression = get_activity_progression_from_df_chris(self._aos_data_df, self.ruleset, self.ruleset_key_idx, self.ruleset_type, rule_progession_out_fname = self._rule_progression_out_fname)
        else:
            print(f"No valid activity generation method. {self._activity_gen_method}")
        t1_stop = process_time() 
        print("Get activity process time: ", t1_stop-t1_start, "s")
    """