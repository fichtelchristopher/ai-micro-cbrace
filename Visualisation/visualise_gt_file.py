'''
Simple script for ground truth visualisation.
'''
from Visualisation.visualisation_parameter_setup import *
from Visualisation.visualisation import plot_default_gt
import pandas as pd
import matplotlib.pyplot as plt

dir = "C:/Users/fichtel/Ottobock SE & Co. KGaA/Deep Learning - General/AI-Pipeline/leipzig/02-Activities-GT"
filename = "DM0104.csv"
f = dir + f"/{filename}"

# Simple visualisation
data_df = pd.read_csv(f)


# data_cols = ["JOINT_ANGLE", LABEL_COL, "JOINT_LOAD"], 
# data_labels = ["ANGLE", LABEL_COL, "JOINT_LOAD"],
# data_scaling_factors = [1, 1/10, 1/100],
# _plot_indices = True

plot_default_gt(data_df, data_cols=["JOINT_LOAD"], data_scaling_factors = [1, 1/100])

plt.show()

print("")

