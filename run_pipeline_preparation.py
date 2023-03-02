'''
Master thesis project of Christopher Fichtel started in Oct 2022
AI in operating ortheses

File created 23 Dec 2022

This file prepares for running the pipeline: run_pipeline
includes ground truth generation, dataset reduction and feature generation (for a feature based approach)
'''
from ground_truth_generation_aos import ground_truth_generation_aos_main
from ground_truth_generation_leipzig import ground_truth_generation_leipzig_main

from data_reduction import reduce_data
from Configs.pipeline_config import *
from Configs.nn_pipeline_config import *
from Configs.shallow_pipeline_config import *
from dataset_generation_shallow import dataset_generation_shallow_main

from Misc.output import *
from optimization import *
from Processing.loading_saving import *
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# a = load_from_pickle("C:/Users/fichtel/Ottobock SE & Co. KGaA/Deep Learning - General/AI-Pipeline/leipzig/03-Features/w16/DM0104 - Kopie.pkl")

# Which steps to run
run_dict = {
    "ground_truth": False,
    "reduce_dataset": False,
    "dataset_generation": True, # includes feature generation
    "feature_selection": False
}

if run_dict["ground_truth"]:
    if DATABASE == "aos":
        ground_truth_generation_aos_main()
    else:
        ground_truth_generation_leipzig_main()

if run_dict["reduce_dataset"]:
    if DATABASE == "aos":
        if DATASET == 1:
            reduce_data(input_dir = ACTIVITIES_FULL_OUTPUT_DIR, output_dir = ACTIVITIES_OUTPUT_DIR, discard_classes=[100], combine_classes=[([200, 210, 220, 230, 240, 300, 310, 320, 330, 340], 100, "level-walking"), ([400, 410, 420, 430, 440], 200, "yielding"), ([500], 300, SWING_EXTENSION_CLASSNAME), ([0, 600], 0, "other")])
        elif DATASET == 2:
            reduce_data(input_dir = ACTIVITIES_FULL_OUTPUT_DIR, output_dir = ACTIVITIES_OUTPUT_DIR, discard_classes=[100], combine_classes=[([200, 220, 230, 240, 300, 320, 330, 340], 100, "level-walking>30"), ([210, 310], 110, "level-walking<30"), ([400, 420, 430, 440], 200, "yielding>30"), ([410], 210, "yielding<30"), ([500], 300, SWING_EXTENSION_CLASSNAME), ([600], 400, "stumbling")])
        else:
            print("Invalid dataset.")
    elif DATABASE == "leipzig":
        if USE_ORIGINAL_LEIPZIG_LABELS:
            reduce_data(input_dir = ACTIVITIES_FULL_OUTPUT_DIR, output_dir = ACTIVITIES_OUTPUT_DIR, discard_classes=[], combine_classes=[([100, 200, 400], 100, "level-walking"), ([300, 500], 200, "yielding"), ([600], 300, SWING_EXTENSION_CLASSNAME)], database = "leipzig")
        else:
            reduce_data(input_dir = ACTIVITIES_FULL_OUTPUT_DIR, output_dir = ACTIVITIES_OUTPUT_DIR, discard_classes=[], combine_classes=[([100, 110, 120, 130, 140, 160, 170, 180, 190], 100, "level-walking"), ([210, 220, 230, 240], 200, "yielding"),  ([0, 350, 400], 0, "other")], database = "leipzig")
    else:
        print(f"Invalid database {DATABASE}")

if run_dict["dataset_generation"]:
    # This is for the feature approach
    dataset_generation_shallow_main(database = DATABASE)