''' 
Master thesis project of Christopher Fichtel started in Oct 2022
AI in operating ortheses

File created 23 Feb 2022

This file applies via loc indices 
'''

from Processing.file_handling import get_leipig_files_from_dir, get_aos_files_from_dir
from Processing.loading_saving import load_from_pickle, save_as_pickle
import os
from Configs.namespaces import LABEL_COL, INDEX_COL
from Visualisation.visualisation import *
from Misc.utils import save_activity_code_type_dict
from Analysis.activity_analysis import dir_activity_analysis
from Analysis.transitions_overview import generate_transition_overview

def apply_mapping(input_filepaths, gt_filepaths, output_dir, mapping_dict, activity_code_type_dict, database):
    '''
    For a list of input filepaths create output filepaths with mapped labels. 

    :param input_filepaths: the paths the dataframes whose labels will be modified. Only the labels are modified, nothing else! 
    :param output_dir: where to save the mapped filespaths and activity code type dict
    :param gt_filepaths: for each input filepath provide the gt filepath. These are the files in 01-Activities-GT-Full. The labels
                        from these files will be mapped to the output (using the loc indices).
    :param mapping_dict: assign the mapping to gt_filepaths and save mapped labels in the input df
    
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_as_pickle(mapping_dict, output_dir + "/mapping_dict.pkl")

    for input_f, gt_f in zip(input_filepaths, gt_filepaths):
        # Find the correct ground truth file
        if not os.path.exists(gt_f):
            if "_reduced.pkl" in gt_f:
                gt_f = gt_f.replace("_reduced.pkl", ".pkl")
            if "_features_normalized.pkl" in gt_f:
                gt_f = gt_f.replace("_features_normalized.pkl", ".pkl")

        if not (os.path.exists(input_f) & (os.path.exists(gt_f))):
            print(f"Could not create mapped file for input {input_f}")
            continue 

        input_df = load_from_pickle(input_f)
        gt_df = load_from_pickle(gt_f)

        assert(LABEL_COL in input_df.columns.values)
        assert(LABEL_COL in gt_df.columns.values)

        if INDEX_COL in input_df.columns.values:
            input_df = input_df.set_index(INDEX_COL)
        if INDEX_COL in gt_df.columns.values:
            gt_df = gt_df.set_index(INDEX_COL)
        
        # we only need the label of the gt file
        gt_df = gt_df.replace({LABEL_COL: mapping_dict})
        gt_df = gt_df[LABEL_COL]
        
        input_df[LABEL_COL].loc[input_df.index] = gt_df.loc[input_df.index]

        assert(set(list(input_df[LABEL_COL].values)).issubset(set(list(activity_dict.keys()))))

        # plot_default_gt(input_df)
        if INDEX_COL not in input_df.columns.values:
            input_df[INDEX_COL] = input_df.index

        save_as_pickle(input_df, output_dir + f"/{os.path.basename(input_f)}")

    save_activity_code_type_dict(activity_code_type_dict, output_dir)

    dir_activity_analysis(output_dir, activity_code_type_dict, database = database)
    
    generate_transition_overview(output_dir, database = database)

    return

if __name__ == "__main__":
    
    database = "aos"
    input_ds = "DS3"
    output_ds = "DS2"

    input_dir = f"D:/AI-Pipeline/{database}/{input_ds}/04-Features-Normalized"
    gt_dir = f"D:/AI-Pipeline/{database}/01-Activities-GT-Full"


    # AOS mappings --> set database to "aos"
    activity_code_type_dict = {
        0: "other",
        100: "level-walking>30",
        110: "level-walking<30",
        200: "yielding>30",
        210: "yielding<30",
        300: SWING_EXTENSION_CLASSNAME,
    }

    mapping_dict_aos_ds3_to_ds1 = {
        0:      0,
        200: 100,
        210: 100,
        220: 100,
        230: 100, 
        240: 100,
        300: 200,
        310: 200,
        320: 200,
        330: 200,
        340: 200,
        400: 300,
        410: 300,
        420: 300,
        430: 300,
        440: 300,
        500: 400,
        600: 500, # stumbling will be other class
    }

    mapping_dict_aos_ds3_to_ds2 = {
        0:      0,
        200: 100,
        210: 100,
        220: 100,
        230: 100, 
        240: 100,
        300: 200,
        310: 200,
        320: 200,
        330: 200,
        340: 200,
        400: 300,
        410: 300,
        420: 300,
        430: 300,
        440: 300,
        500: 400,
        600: 500, # stumbling will be other class
    }
    

    # input_dir = f"C:/Users/fichtel/Ottobock SE & Co. KGaA/Deep Learning - General/AI-Pipeline/{database}/{input_ds}/04-Features-Normalized"
    # gt_dir = f"C:/Users/fichtel/Ottobock SE & Co. KGaA/Deep Learning - General/AI-Pipeline/{database}/01-Activities-GT-Full"
    input_dir = f"D:/AI-Pipeline/{database}/{input_ds}/02-Activities-GT"
    gt_dir = f"D:/AI-Pipeline/{database}/01-Activities-GT-Full"
    
    if database == "leipzig":
        input_files = get_leipig_files_from_dir(input_dir)
    else:
        input_files = get_aos_files_from_dir(input_dir)

    gt_files = [gt_dir + f"/{os.path.basename(f)}" for f in input_files]

    output_dir = input_dir.replace(input_ds, output_ds)
    # output_dir = "C:/Users/fichtel/Ottobock SE & Co. KGaA/Deep Learning - General/AI-Pipeline/leipzig/DS2/02-Activities-GT"

    apply_mapping(input_files, gt_files, output_dir, mapping_dict_aos_ds3_to_ds2, activity_code_type_dict, database)

    print("Done mapping")
    

    # '''
    # Leipzig Mappings --> set database to leipzig

    activity_code_type_dict_leipzig = {
        0: "other",
        100: "level-walking",
        200: "yielding",
        300: SWING_EXTENSION_CLASSNAME, 
        400: "ramp-ascent",
        500: "stair-ascent"        
    }

    mapping_dict_leipzig = {
        0:      0,
        110:    100,
        120:    100,
        130:    100,
        140:    100,
        160:    400,
        170:    400,
        180:    400,
        190:    400,    
        210:    200,
        220:    200,
        230:    200,
        240:    200,
        300:    300,
        350:    500,
        400:    500,
    }

    input_files = get_leipig_files_from_dir(input_dir)

    gt_files = [gt_dir + f"/{os.path.basename(f)}" for f in input_files]

    output_dir = input_dir.replace(input_ds, output_ds)
    # output_dir = "C:/Users/fichtel/Ottobock SE & Co. KGaA/Deep Learning - General/AI-Pipeline/leipzig/DS2/02-Activities-GT"

    apply_mapping(input_files, gt_files, output_dir, mapping_dict_leipzig, activity_code_type_dict_leipzig, database)
    
    # '''