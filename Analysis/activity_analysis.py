import pandas as pd
import numpy as np 
import os 
import glob 

import matplotlib.pyplot as plt

from Configs.namespaces import *
from Configs.pipeline_config import DISCARD_CLASS_TYPE, DISCARD_CLASS_CODE
from Visualisation.visualisation import * 
from Misc.utils import load_from_pickle, save_as_pickle

from Processing.file_handling import get_leipig_files_from_dir, get_aos_files_from_dir

def dir_activity_analysis(data_dir, activity_code_type_dict = ACTIVITY_CODE_TYPE_DICT_CHRIS, database = "aos"):
    '''
    Give an overview about frequency of activity classes for individual files in data dir but also accumulated.
    Will 

    :param database: "aos" or "leipzig"
    '''
    if database == "aos":
        files = get_aos_files_from_dir(data_dir)
    elif database =="leipzig":
        files =  get_leipig_files_from_dir(data_dir)
    else:
        print(f"Unknown database {database}.")

    output_dir = os.path.join(data_dir, "activity overview")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    patient_label_dict_accum = {}

    activity_overview_df = pd.DataFrame()

    for c, f in enumerate(files):
        patient = os.path.basename(f).replace(".pkl", "")
        # patient = patient.replace("_gt_reduced", "")
        patient = patient.replace("_gt", "")

        output_fname = output_dir + f"/{patient}_activity_label_overview.png"
        # data_df = pd.read_csv(f)
        data_df = load_from_pickle(f)
        patient_label_dict = activity_analysis(data_df, output_fname, activity_code_type_dict=activity_code_type_dict)
        
        if c == 0:
            patient_label_dict_accum = patient_label_dict
        else:
            for key, item in patient_label_dict.items():
            
                if key in patient_label_dict_accum.keys():
                    patient_label_dict_accum[key] = patient_label_dict_accum[key] + patient_label_dict[key]
                else:
                    patient_label_dict_accum[key] = patient_label_dict[key]

        # For the csv overview file
        patient_label_dict_k = {}
        patient_label_dict_k["file"] = os.path.basename(f)
        if "TIME" in data_df.columns.values:
            patient_label_dict_k["duration[min]"] = np.round(data_df["TIME"].values[-1] / 60, 2)
        else:
            patient_label_dict_k["duration[min]"] = 0
        for code, val in patient_label_dict.items():
            patient_label_dict_k[activity_code_type_dict[code]] = val
        activity_overview_df = activity_overview_df.append(patient_label_dict_k, ignore_index = True)
    
    output_fname = output_dir + f"/{database}_directory_activity_label_overview.png"
    labels = [activity_code_type_dict[v[0]] for v in patient_label_dict_accum.items()]

    total_samples = 0
    for key, num in patient_label_dict_accum.items():
        total_samples += num

    visualize_dict_as_pie_chart(patient_label_dict_accum, output_file=output_fname, labels = labels, title=f"accumulated activity overview\ntotal of {total_samples} samples")# for input dir\n{data_dir}")
    plt.clf()

    # Create csv
    activity_overview_df.to_csv(output_fname.replace(".png", ".csv"), index = False)

def get_label_distr_dict(activity_code_type_dict:dict, activity):
    '''
    For each key in activitiy code type dict, count the number of occurences
    in the activity (list or array) 

    :param activity: 1D ist or array of activity values e.g. [0, 0, 0, 100, 100, 0, 0, 300, 300, 300, ... ]
    '''
    activity = list(activity)
    
    label_dict = {}
    for key in list(activity_code_type_dict.keys()):

        label_dict[key] =  activity.count(key)

    return label_dict

def get_label_distr_dict_from_activity_list(activity):
    '''
    For the set of activities in the activity list/array return a dictionary with the occurences.
    '''
    activity = list(activity)
    keys = list(set(activity))
    
    label_distribution_dict = {}
    for key in keys:

        label_distribution_dict[key] = activity.count(key)

    label_distribution_dict = dict(sorted(label_distribution_dict.items(), key=lambda item: item[1]))

    return label_distribution_dict

def activity_analysis(aos_data_df, output_fname, activity_code_type_dict = ACTIVITY_CODE_TYPE_DICT_CHRIS, ignore_discard_class = True):
    '''
    Give an overview about frequency of activity classes for a given dataframe.
    '''
    activity = list(aos_data_df[LABEL_COL].values)

    if "TIME" in aos_data_df.columns.values:
        duration_in_hours = np.round(aos_data_df["TIME"].values[-1] / 3600, 2)
    else:
        duration_in_hours = 0

    if ignore_discard_class:
        activity_code_type_dict = {key:val for key, val in activity_code_type_dict.items() if val != DISCARD_CLASS_TYPE}

    label_dict = get_label_distr_dict(activity_code_type_dict, activity)

    # Visualize as pie chart
    patient = os.path.basename(output_fname).split(".")[0]
    title = f"{len(activity)} samples for\n{patient}"
    labels = [activity_code_type_dict[v[0]] for v in label_dict.items()]
    visualize_dict_as_pie_chart(label_dict, output_file=output_fname, labels = labels, title=title)
    plt.clf()

    full_label_dict = label_dict.copy()

    # Visualize without other class 
    del label_dict[0]
    title = f"{len(activity)} samples for\n{patient}.\nTotal Duration: {duration_in_hours} hours"
    labels = [activity_code_type_dict[v[0]] for v in label_dict.items()]
    visualize_dict_as_pie_chart(label_dict, output_file=output_fname.replace(".png", "_without_0_class.png"), labels = labels, title=title)
    plt.clf()

    return full_label_dict