import pandas as pd
import numpy as np 
import os 
import glob 

import matplotlib.pyplot as plt

from Configs.namespaces import *
from Configs.pipeline_config import DISCARD_CLASS_TYPE, DISCARD_CLASS_CODE
from Visualisation.visualisation import * 
from Analysis.activity_analysis import * 


def main():
    train_dir = "C:/Users/fichtel/Ottobock SE & Co. KGaA/Deep Learning - General/Beispieldaten_AOS\AI-Pipeline/04-Train-Val-Normalized-Reduced/TRAIN_0.7"

    train_files = glob.glob(train_dir + "/*features_train.csv")
    # or
    # train_files = []

    label_dict = {}
    for key in list(ACTIVITY_CODE_TYPE_DICT_CHRIS.keys()):
        label_dict[key] = 0

    total_count = 0

    for f in train_files:

        total_count_patient = 0

        label_dict_f = {}
        for key in list(ACTIVITY_CODE_TYPE_DICT_CHRIS.keys()):
            label_dict[key] = 0

        train_df = pd.read_csv(f)

        labels = list(train_df["ACTIVITY"].values)

        for key in list(ACTIVITY_CODE_TYPE_DICT_CHRIS.keys()):

            label_dict[key] = label_dict[key] + labels.count(key)

            num = labels.count(key)
            label_dict_f[key] = num

            total_count += num
            total_count_patient += num 

        # Visualize as pie chart
        labels = [ACTIVITY_CODE_TYPE_DICT_CHRIS[v[0]] for v in label_dict.items()]
        patient = os.path.basename(f).replace("_gt_features_train.csv", "")
        output_fname = train_dir + f"/{patient}_train_label_overview.png"
        title = f"{total_count_patient} samples for {patient}"
        visualize_dict_as_pie_chart(label_dict_f, output_file=output_fname, labels = labels, title=title)
        plt.clf()

    output_fname = train_dir + "/train_label_overview.png"
    labels = [ACTIVITY_CODE_TYPE_DICT_CHRIS[v[0]] for v in label_dict.items()]
    title = f"{total_count} samples"
    visualize_dict_as_pie_chart(label_dict, output_file=output_fname, labels = labels,title=title)
    plt.clf()

if __name__ == "__main__":
    main()





