''' 
Master thesis project of Christopher Fichtel started in Oct 2022
AI in operating ortheses

File created  12.02.2023

Ground Truth Generation using manual labels of RA, RD, SA, SD and LW for data acquired in the Leipzig fair 2018.
'''

from Misc.utils import *
from Processing.loading_saving import load_from_pickle, save_as_pickle
from Processing.loading_saving import *
from AOS_SD_analysis.AOS_Subject import * 
from Processing.preprocessing import *
from Processing.file_handling import * 
from AOS_SD_analysis.AOS_Rule_Set_Parser import * 
from Misc.utils import compress_state_progression
from Configs.pipeline_config import *
from Visualisation.visualisation import *
from Visualisation.namespaces_visualisation import *
from Gait.steps_utils import *

from data_reduction import reduce_data

from Analysis.activity_analysis import *

def add_activity_leipzig(aos_data_df, labels_indices_dict, activity_type_code_dict):
    '''
    
    '''
    activity_stream = np.zeros(len(aos_data_df))
    ic_indices_stream = np.zeros(len(aos_data_df))
    swr_indices_stream = np.zeros(len(aos_data_df))

    assert(SWING_EXTENSION_CLASSNAME in activity_type_code_dict)

    for step_type, indices in labels_indices_dict.items():

        step_code = activity_type_code_dict[step_type]

        steps = detect_swr_without_ruleset(aos_data_df, indices)

        for step_start_idx, swr_idx, step_stop_idx in steps: 

            activity_stream[step_start_idx:swr_idx] = step_code
            activity_stream[swr_idx:step_stop_idx] = activity_type_code_dict[SWING_EXTENSION_CLASSNAME]

            ic_indices_stream[step_start_idx] = 1
            ic_indices_stream[step_stop_idx] = 1
            swr_indices_stream[swr_idx] = 1

    aos_data_df[LABEL_COL] = activity_stream
    aos_data_df[IC_COL] = ic_indices_stream
    aos_data_df[SWR_COL]  = swr_indices_stream

    return aos_data_df
    
def add_activity_leipzig_from_label_file(aos_data_df, label_file, activity_code_type_dict = ACTIVITY_CODE_TYPE_DICT_LEIPZIG_ADAPTED, walking_classes = WALKING_CLASSES_LEIPZIG):
    '''
    From a label file add the activity column.
    
    :param walking_classes: in conjunction with walking ranges, for the walking classes define ranges to further split a walking class
            e.g. instead of having only one level walking class, by supplying a list of ranges e.g. [(0, 20), (20, 60), (60, 100)]
            level walking class will be split in 3 subclasses where the range indicates the range in which the maximum
            knee angle is for a given step
    :param walking_classes_ranges: if empty list --> do not add ranges
    '''
    activity_stream = np.zeros(len(aos_data_df))
    ic_indices_stream = np.zeros(len(aos_data_df))
    swr_indices_stream = np.zeros(len(aos_data_df))

    gt_indices = load_leipzig_gt_from_label_file(label_file)

    for x, label_code in gt_indices:
        x = int(x) - 1
        if x < 0:
            x = 0
        label_code = int(label_code)
        if (label_code < 100) & (label_code > 0):
            label_code *= 10 

        if activity_code_type_dict[label_code] in walking_classes:
            ic_indices_stream[x] = 1
        if activity_code_type_dict[label_code] == SWING_EXTENSION_CLASSNAME:
            swr_indices_stream[x] = 1
        
        label_code_prev = activity_stream[x-1]
        if activity_code_type_dict[label_code_prev] == SWING_EXTENSION_CLASSNAME:
            ic_indices_stream[x] = 1
            
        assert((x >= 0) & (x < len(activity_stream)))
        
        activity_stream[x:] = label_code

    aos_data_df[LABEL_COL] = activity_stream
    aos_data_df[IC_COL] = ic_indices_stream
    aos_data_df[SWR_COL]  = swr_indices_stream
    
    return aos_data_df


def add_walking_ranges(data_df, walking_classes, walking_classes_ranges):
    
    activity_stream = data_df[LABEL_COL].values
    activities, activity_start_indices = compress_state_progression(activity_stream)
    activities = list(np.array(activities).astype(int))
    activities.append(-1)
    activity_start_indices = list(activity_start_indices)
    activity_start_indices.append(len(data_df))

    for activity, start_index, end_index in zip(activities[:-1], activity_start_indices[:-1], activity_start_indices[1:]):
    
        if activity in walking_classes:

            max_joint_angle = max(data_df.iloc[start_index:end_index]["JOINT_ANGLE"].values)
        
            for ind, (r_min, r_max) in enumerate(walking_classes_ranges):
                if (max_joint_angle >= r_min) & (max_joint_angle < r_max):
                    data_df[LABEL_COL].iloc[start_index:end_index] = data_df[LABEL_COL][start_index] + (ind+1)*10

    return data_df

def ground_truth_generation_leipzig_main(raw_input_data_dir = DATA_DIR_LEIPZIG,
                                        activities_full_output_dir = ACTIVITIES_FULL_OUTPUT_DIR,
                                        use_original_leipzig_labels = USE_ORIGINAL_LEIPZIG_LABELS,
                                        walking_classes = WALKING_CLASSES_LEIPZIG,
                                        walking_classes_ranges = WALKING_CLASSES_RANGES):

    if use_original_leipzig_labels:
        # The labels with ramp ascend, ramp descend etc.
        activity_code_type_dict = ACTIVITY_CODE_TYPE_DICT_LEIPZIG_ORIGINAL
        activity_type_code_dict = ACTIVITY_TYPE_CODE_DICT_LEIPZIG_ORIGINAL
        save_activity_code_type_dict(activity_code_type_dict, activities_full_output_dir, out_basename = "activity_code_type_dict")
    else:
        activity_code_type_dict_in  = ACTIVITY_CODE_TYPE_DICT_LEIPZIG_ADAPTED
        activity_type_code_dict_in     = ACTIVITY_TYPE_CODE_DICT_LEIPZIG_ADAPTED
        activity_code_type_dict  = ACTIVITY_CODE_TYPE_DICT_LEIPZIG_ADAPTED_OUT
        activity_type_code_dict     = ACTIVITY_TYPE_CODE_DICT_LEIPZIG_ADAPTED_OUT

        walking_classes_codes = [activity_type_code_dict_in[walking_type] for walking_type in walking_classes]
        save_activity_code_type_dict(activity_code_type_dict, activities_full_output_dir, out_basename = "activity_code_type_dict")

    data_files_leipzig_dict = get_data_files_leipzig_dict(raw_input_data_dir)

    for subject_key, subject_files in data_files_leipzig_dict.items():
        # The subject output file from multiple input files
        subject_df_out_fname = get_subject_df_out_fname_leipzig(activities_full_output_dir, subject_files[0], subject_key)

        if os.path.exists(subject_df_out_fname):
            continue

        aos_data_df_comb = pd.DataFrame()

        for idx, f in enumerate(subject_files):

            data_df = load_data_file(f) # supports mat, csv, txt file
            data_df = preprocess_df(data_df, knee_lever = KNEE_LEVER, aos_data = False) 

            if use_original_leipzig_labels:
                # Get the labels dict containing the step indices ranges (label corresponds to the step type e.g. "RA", "LW", ...)
                label_files = get_leipzig_label_files(f)
                if len(label_files) == 0:
                    continue
                labels_indices_dict = get_labels_indices_dict(label_files)

                data_df = add_activity_leipzig(data_df, labels_indices_dict, activity_type_code_dict)
            else: 
                gtLabels_filename = os.path.basename(f).replace(".txt", "_gtLabels.mat")
                label_file = str(Path(f).parent.parent) + f"/{gtLabels_filename}"
                if os.path.exists(label_file):
                    data_df = add_activity_leipzig_from_label_file(data_df, label_file, activity_code_type_dict_in, walking_classes)
                    data_df[LABEL_FILE_COL] = os.path.basename(label_file)

                    data_df = add_walking_ranges(data_df, walking_classes_codes, walking_classes_ranges)

                    # Save ground truth as png output for fast visualisation 
                    subject_gt_output_plots_dir = activities_full_output_dir + f"/GT Plots/{subject_key}"
                    if not os.path.exists(subject_gt_output_plots_dir):
                        os.makedirs(subject_gt_output_plots_dir)
                    subject_plot_basename = os.path.basename(gtLabels_filename).replace(".mat", ".png")
                    subject_gt_output_plot_f = subject_gt_output_plots_dir + f"/{subject_plot_basename}"
                    plot_default_gt(data_df, _plot_indices = False)
                    plt.title(f"{subject_key}_{gtLabels_filename}", fontsize = 11)
                    plt.tight_layout()
                    plt.savefig(subject_gt_output_plot_f, dpi = 300)
                    plt.clf()
                else: 
                    continue

            data_df[ACTIVITY_IDX_COL] = idx

            if idx == 0:
                aos_data_df_comb = data_df            
            else: 
                aos_data_df_comb = aos_data_df_comb.append(data_df, ignore_index = True)

        # Check if ground truth label files where found
        # if not --> aos_data_df_comb length will be zero
        if len(aos_data_df_comb) == 0:
            continue

        # Preprocess the data. Add time column, knee moment, delete "unneccessary" columns in order to save memory
        aos_data_df_comb[INDEX_COL] = np.arange(len(aos_data_df_comb))

        # Save as csv
        save_as_pickle(aos_data_df_comb, subject_df_out_fname)

        visualize = False
        if visualize:
            matplotlib.use("TkAgg")
            plot_default_gt(aos_data_df_comb, _plot_indices = True)
            plt.plot(aos_data_df_comb[ACTIVITY_IDX_COL])
            n_input_gt_files = len(list(set(list(aos_data_df_comb[ACTIVITY_IDX_COL].values))))
            plt.title(f"{subject_key}\tUsing {n_input_gt_files} input gt files.")
            plt.show()
            plt.clf()
            matplotlib.use("Agg")
    
    dir_activity_analysis(activities_full_output_dir, activity_code_type_dict=activity_code_type_dict, database = "leipzig")
    save_activity_code_type_dict(activity_type_code_dict, activities_full_output_dir, out_basename = "leipzig_activity_code_type_dict")
            
if __name__ == "__main__":
    ground_truth_generation_leipzig_main(use_original_leipzig_labels = False)