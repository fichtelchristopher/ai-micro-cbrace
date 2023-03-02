''' 
Master thesis project of Christopher Fichtel started in Oct 2022
AI in operating ortheses

File created 06 Oct 2022

Code base for various visualisations.
'''
from Visualisation.visualisation_parameter_setup import * 
from Visualisation.namespaces_visualisation import *
import matplotlib
matplotlib.use('Agg') # backend = matplotlib.rcParams["backend"]
import matplotlib.pyplot as plt
import numpy as np
import math

from sklearn.metrics import ConfusionMatrixDisplay

def plot_signals_from_df(data_df, data_cols, data_labels = [], data_scaling_factors = [], data_colors = [], range = None, x = None, provide_labels = True, title = None, linestyles = []): 
    '''
    :param data_df: the dataframe to plot from, must contain all the columns described in data_cols 
    :param range:   None --> whole df 
                    otherwise indicate via (start_idx, end_idx) to only plot a certain range
    :param provide_labels: if False --> dont submit a label when calling plt.plot()
                            useful when calling the function multiple times with same label --> legend would accumulate these

    '''
    assert(set(data_cols).issubset(set(list(data_df.columns.values))))

    if type(x) is type(None):
        x = np.arange(len(data_df)) # plot x axis as samples if no axis provided 
    else:
        assert(len(x) == len(data_df))
        
    if range: 
        assert(len(range) == 2)
        data_df = data_df.iloc[range[0]:range[1]]

    if len(data_labels) != len(data_cols):
        data_labels = data_cols
    
    if len(linestyles) != len(data_cols):
        if PREDICTION_COL in data_labels:
            linestyles = ["-"  if s != PREDICTION_COL else "--" for s in data_cols]
        else:
            linestyles = ["-"] * len(data_cols)
    
    if len(data_scaling_factors) != len(data_cols):
        print("Setting every scaling factor to 1.")
        data_scaling_factors = list(np.ones(len(data_cols)))
    
    provided_color_list = True
    if len(data_colors) != len(data_cols):
        data_colors = [None] * len(data_cols)
        provided_color_list = False


    for data_col, data_label, data_scaling_factor, data_color, linestyle in zip(data_cols, data_labels, data_scaling_factors, data_colors, linestyles):
        if not provide_labels:
            data_label = None
        
        if provided_color_list:
            plt.plot(x, data_df[data_col].values * data_scaling_factor, label = data_label, color = data_color, linestyle = linestyle)
        else:
            if data_col in SIGNAL_COLOR_DICT.keys():
                data_color = SIGNAL_COLOR_DICT[data_col]
            plt.plot(x, data_df[data_col].values * data_scaling_factor, label = data_label, color = data_color, linestyle = linestyle)

    if type(title) != type(None):
        plt.title(title, fontsize = 12)
    
    plt.legend()

    plt.grid(visible = True, axis = "y")

    plt.tight_layout()

def plot_indices(ic_indices, ymin = -100, ymax = 100, color = "blue", alpha = 0.5, label = IC_COL):
    '''
    Plots at the corresponding initial contact indices as a vertical line.

    :param ic_indices: list of indices (i.e. x positions)
    '''
    plt.vlines(x = ic_indices, ymin=ymin, ymax = ymax, color = color, alpha = alpha, label = label)

def plot_cutting_areas(df, data_cols, cut_activity_ranges, data_scaling_factors = None): 
    '''
    
    :param df:
    :param data_cols:

    '''
    if type(data_scaling_factors) == type(None):
        data_scaling_factors = len(np.ones(shape=(len(data_cols))))
    assert(len(data_cols) == len(data_scaling_factors))

    plot_signals_from_df(df, data_cols=data_cols, data_scaling_factors=data_scaling_factors)
    for c in cut_activity_ranges:    
        plot_range(c[0], c[1], ymin = 0, ymax = 500, alpha = 0.5, color = "red")


def plot_range(start_idx, end_idx, ymin = 0, ymax = 100, label = None, color = None, alpha = 0.5, show_label = False, plot_step_edges = True):
    '''
    Spans up a certain color in the range of start_idx and end_idx. Note that the plot already should exist.
    Exeamplary "application": plot step ranges 

    :param start_idx:   where a step starts
    :param end_idx:     where a step stops    
    :param label:       has to be in STEP_LABEL_COLOR_DICT keys if no Color submitted
    :color:             if None --> make sure the label is in STEP_LABEL_COLOR_DICT's keys (see namespaces_visualisation.py)
    :param show_label:  whether or not a label should be added. If you plot multiple steps from the same type make sure
                        to set show_label to True only once! 
    :param plot_step_edges: if True --> plot solid line at start and end idx

    '''
    if color is None:
        if label:
            assert(label in STEP_LABEL_COLOR_DICT.keys())
            color = STEP_LABEL_COLOR_DICT[label]

    if show_label:
        plt.axvspan(start_idx, end_idx, facecolor = color, alpha=alpha, label = label, ymin = ymin, ymax=ymax)
    else:
        plt.axvspan(start_idx, end_idx, facecolor = color, alpha=alpha, ymin = ymin, ymax=ymax)

    if plot_step_edges:
        plt.axvline(x = start_idx,  color = color)
        plt.axvline(x = end_idx,    color = color)

def plot_data_w_steps(data_df, labels_indices_dict, data_cols, data_labels, data_colors, data_scaling_factors, title = ""):
    '''
    Plots signals from the dataframe determined by "data_cols" and the steps described by labels_indices_dict

    :param df: input dataframe that contains the data columns described by data_cols
    :param labels_indices_dict: dictionary with step label as key and list of tuples (step_start_idx, step_end_idx) as items
                                e.g. 
                                labels_indices_dict = 
                                {
                                    "RA": [(start_idx1, end_idx1), (start_idx2, end_idx2), (...), ...],
                                    "RD": [(...), (...), ...]
                                    ...
                                }
    :param data_cols:               list of lenght x
    :param data_labels:             list of lenght x, labels for data columns 
    :param data_colors:             list of lenght x, colors for data columns
    :param data_scaling_factors:    list of lenght x, scaling factors for signals 
                                    signal_plot = scaling_factor * signal 
    :title: optional plot tile
    '''
    assert((len(data_cols) == len(data_labels)) & (len(data_labels) == len(data_colors)) & (len(data_scaling_factors) == len(data_colors)))
    
    plot_signals_from_df(data_df, data_cols, data_labels = data_labels, data_scaling_factors=data_scaling_factors) #, data_colors = data_colors)

    for step_label, step_indices in labels_indices_dict.items():
        show_label = True
        for (start_idx, end_idx) in step_indices:
            plot_range(start_idx, end_idx, label = step_label, show_label = show_label, alpha = 0.3)
            show_label = False

    plt.xlabel("Samples (f sampling = 100Hz)")
    plt.title(title)
    plt.legend()


def plot_typical_steps(typical_step_df, signals, interpolated_step_dfs = None, title = "", output_file = None):
    '''
    :param typical_step_dfs: dataframe, for each signal it has the column name f"{signal}_mean" and f"{signal}_std"
    :param signals: list of signals, if interpolated step                 
    :param interpolated_step_dfs_dict:
    .param output_file: if provided --> save figure
    '''
    ncols = 3
    nrows = int(math.ceil(len(signals) / 3))
    fig, axes = plt.subplots(ncols = ncols, nrows=nrows,  figsize = (ncols * 5, nrows * 5))

    fig.suptitle(title, fontsize=20)

    for ind, ax in enumerate(fig.axes): 
        if ind >= len(signals):
            break
        signal = signals[ind] 

        interpolated_step_signals = [step_df[signal].values for step_df in interpolated_step_dfs]

        typical_step_mean   = typical_step_df[signal + TYPICAL_STEP_MEAN_STR]
        typical_step_std    = typical_step_df[signal + TYPICAL_STEP_STD_STR]

        plot_typical_step(typical_step_mean, typical_step_std, interpolated_step_signals, signal = signal, ax = ax)

    fig.tight_layout(pad = 2.0)

    if output_file is not None:
        plt.savefig(output_file) #, dpi = 100)
        plt.close(fig)
    else:
        plt.show()

def plot_typical_step(typical_step_mean, typical_step_std, individual_steps_list = [], title = None, signal = "", color = "blue", show = False, ax = None, xlabel = "Gait cycle [%]"):
    '''
    
    :param typical_step_mean:   array or list of values, avg of a step usually len 101
    :param typical_step_std:    array or list of values, std of a step (same length as typical_step_mean)
    :param individual_steps_list:   list of steps, individual steps have to have same length as typical_step_mean and std
                                    if provided --> plot them as gray lines with lower alpha
    :param step_label: to which step the plot belongs
    :param signal: which signal for the step is plotted
    :param color: color for mean and std
    :param show: whether or not to call plt.show()
    
    '''
    assert(len(typical_step_mean) == len(typical_step_std))

    if ax is None:
        fig, ax = plt.subplots()

    x = np.arange(0, len(typical_step_mean))

    for individual_step in individual_steps_list:
        ax.plot(x, individual_step, color = "grey", alpha = 0.4)

    ax.plot(x, typical_step_mean, c=color, linewidth=2.5)
    ax.fill_between(x, typical_step_mean-typical_step_std, typical_step_mean+typical_step_std, alpha=0.6, facecolor=color)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(signal, fontsize=12)
    ax.grid(visible = True, axis = "both")

    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

    if title:
        fig.suptitle(title, fontsize = 14)
    
    if show: 
        plt.show()


def visualize_confusion_matrix(cm, display_labels = list(ACTIVITY_TYPE_CODE_DICT_CHRIS.keys()), title = "", out_save_fname = ""):

    display_labels = [l.replace("-", "\n") for l in display_labels]


    disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
    disp.plot()
    # fig = disp.ax_.get_figure() 
    # fig.set_figwidth(3)
    # fig.set_figheight(3)
    plt.title(title, fontsize = 8)
    plt.xlabel("Predicted Label", fontsize=8, fontweight='bold')
    plt.ylabel("True Label", fontsize=8, fontweight='bold')

    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    plt.tight_layout()

    if len(out_save_fname) > 0:
        plt.savefig(out_save_fname, dpi = 300, figsize = (12, 12))
    plt.close("all")
    return

def visualize_dict_as_boxplot(transition_dict, output_file = None):
    '''
    :param df: describes a transitions
    '''
    transition_dict =  {k:transition_dict[k] for k in sorted(transition_dict)}

    labels = list(transition_dict.keys())
    labels = [l.replace("-->", "\u2192") for l in labels]

    indices = np.arange(len(transition_dict))
    height = np.array(list(transition_dict.values())).astype(int)

    fig, ax = plt.subplots()
    bars = ax.barh(labels, height)

    for bars in ax.containers:
        ax.bar_label(bars, size = 6)

    ax.set_yticks(indices) # + width/2)
    ax.set_yticklabels(labels, minor=False, fontsize = 5)
    plt.title('Overview of transitions')
    plt.tight_layout()

    if type(output_file) != type(None):
        plt.savefig(output_file, dpi = 300)
    else:
        plt.show()

    return 

def visualize_dict_as_pie_chart(dictionary, output_file = None, labels = None, title = ""):
    '''
    :param output_file: if submitted --> save figure
    :param labels: if labels is none, infer them from the dictionary keys
    
    '''

    if type(labels) == type(None):
        labels = list(dictionary.keys())

    values = [v[1] for v in dictionary.items()]

    num_total = sum(dictionary.values())

    def absolute_value(val):
        a  = np.round(val/100.*num_total, 0)
        return int(a)

    # fig, ax = plt.subplots()

    colors = get_colors_from_labels(labels)
    plt.pie(values, colors=colors) #, autopct='%.2f')
    
    # plt.legend(patches, labels, loc="best")
    values_percentage = [np.round(v/num_total*100, 2) for v in values] 
    legend_labels = [f"{l}: {v} ({v_percentage}%)" for l, v, v_percentage in zip(labels, values, values_percentage)]
    plt.legend(legend_labels, loc = "best", bbox_to_anchor=(1, 1))
    plt.axis()
    plt.title(title)
    plt.tight_layout()
    
    if type(output_file) != type(None):     
        plt.savefig(output_file)

def get_colors_from_labels(labels):
    '''
    Get colors according to label in ACTIVITY_LABEL_COLOR_DICT in Visualisation.namespaces_visualisation.py
    if not every color is defiend by ACTIVITY_LABEL_COLOR_DICT label, None will be returned for the colors
    object, meaning matplotlib will use its own color coding.
    :param labels: list of (activity) labels as string. Implemented activities saved in ACTIVITY_LABEL_COLOR_DICT
    '''
    colors = []
    for l in labels: 
        if l in ACTIVITY_LABEL_COLOR_DICT.keys():
            colors.append(ACTIVITY_LABEL_COLOR_DICT[l])
        else:
            colors = None
            break
    
    return colors
    
def visualise_training(model, model_out_fname):
    '''
    :param model: a trained model - after a model has been trained one can call model.history 
    '''
    history = model.history.history
    for key in history.keys():
        values = history[key]

        plt.plot(values)
        plt.title(f"Train {key}")
        plt.xlabel("epochs")
        plt.ylabel(key)
        if key in ["loss"]:
            plt.title(f"Lowest train {key} = {np.min(values)}")
        if key in ["accuracy"]:
            plt.title(f"Highest train {key} = {np.max(values)}")
        plt.savefig(model_out_fname.replace(".h5", f"_{key}.png"))
        plt.clf()
        plt.close("all")

        print(f"Visualised training {key} for model {model_out_fname}.")

    print("")


def plot_default_gt(aos_data_df, 
                    data_cols = ["JOINT_ANGLE", LABEL_COL, "JOINT_LOAD"], 
                    data_labels = ["ANGLE", LABEL_COL, "JOINT_LOAD"],
                    data_scaling_factors = [1, 1/10, 1/100],
                    _plot_indices = True):
    '''
    Plot defaults signals from the ground truth file generated from ground_truth_generation
    '''
    matplotlib.use("TkAgg")
    x = aos_data_df.index.values
    plot_signals_from_df(aos_data_df, data_cols = data_cols, data_labels = data_labels, data_scaling_factors = data_scaling_factors, x = x)
    
    if _plot_indices:
        ic_indices = aos_data_df[aos_data_df[IC_COL] == 1].index.values
        swr_indices = aos_data_df[aos_data_df[SWR_COL] == 1].index.values
        plot_indices(ic_indices, color = SIGNAL_COLOR_DICT[IC_COL], alpha = 0.8, ymax = 20, ymin = -20, label = IC_COL)
        plot_indices(swr_indices, color = SIGNAL_COLOR_DICT[SWR_COL], alpha = 0.8, ymax = 20, ymin = -20, label = SWR_COL)
