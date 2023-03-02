from Configs.namespaces import * 

# A dictionary assigning a color to the 5 labels defined for the leipzig data.
STEP_LABEL_COLOR_DICT = {
    "RA": "red",  # ramp ascent
    "RD": "orange",     # ramp descent
    "SA": "green",      # stair ascent
    "SD": "purple",     # stair descent
    "LW": "blue"       # level walking (even and uneven ground brought together i.e. LWu, LWe, LW_even all LW now)
}
# A dictionary assigning a string (e.g. used for titles) to the 5 labels defined for the leipzig data.
STEP_LABEL_DICT_LEIPZIG = {
    "RA": "ramp ascent",  # ramp ascent
    "RD": "ramp descent",     # ramp descent
    "SA": "stair ascent",      # stair ascent
    "SD": "stair descent",     # stair descent
    "LW": "level walking"       # level walking (even and uneven ground brought together i.e. LWu, LWe, LW_even all LW now)
}

ACTIVITY_LABEL_COLOR_DICT = {
    "other": "green",
    "level-walking": "blue",
    "walking-no-flexion": "cornflowerblue",
    "walking-w-flexion": "midnightblue",
    "yielding": "orange",
    SWING_EXTENSION_CLASSNAME: "red",
    "sitting": "purple",
    "stumbling": "brown",

    "level-walking<30": "steelblue",
    "level-walking>30": "dodgerblue",
    "yielding<30": "darkorange", 
    "yielding>30": "goldenrod"
}

# Creating typical step plots
TYPICAL_STEP_MEAN_STR = "_mean"
TYPICAL_STEP_STD_STR  = "_std"

# Color Definitions for signals - the keys are the same as the column names in the dataframe generated when creating an AOS_subject (or also the input data)
SIGNAL_COLOR_DICT = {
                    LABEL_COL:              "limegreen",
                    PREDICTION_COL:         "r",

                    TOTAL_ACC_COL:          "dodgerblue",
                    "JOINT_LOAD":           "darkmagenta",
                    "KNEE_MOMENT":          "purple",
                    "JOINT_ANGLE":          "darkorange",
                    "JOINT_ANGLE_VELOCITY": "brown",
                    "RI_RULID1":            "grey",
                    "RI_RULID2":            "lightgrey",
                    IC_COL:                 "darkmagenta",
                    SWR_COL:                "orchid"
                    }

METRIC_COLOR_DICT = {
    "ACCURACY": "midnightblue",
    "VAL_ACCURACY": "lightskyblue",
    "F1_SCORE": "green",
    "VAL_F1_SCORE": "lightgreen",
    "LOSS": "red",
    "VAL_LOSS": "lightcoral",
    "LR": "orange"
}