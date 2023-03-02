# from training_nn import * 
# from prediction_nn import *
# from evaluation_nn import *
from training import *
from Configs.pipeline_config import * 
from Models.neural_networks import * 
from Misc.ai_utils import * 
from Misc.utils import print_platform_specifications
from time import process_time
from feature_selection import *
from Visualisation.namespaces_visualisation import METRIC_COLOR_DICT
from Misc.output import *
from prediction import combine_probability_predictions, probabilities_from_path_to_classification, get_prediction_nn, combine_predictions_gt, get_probabilities_nn, probabilities_df_to_classification
from Misc.dataset_utils import get_indices_from_file, get_loc_indices_list_from_files
from Misc.utils import remove_empty_dir, clean_up_directory, move_files
from evaluation import *
import copy

from Models.neural_networks_configurations import get_network_config
from Models.shallow_classifier_configurations import get_shallow_classifier_config

import pickle as pkl
import sys

class Classifier():

    def __init__(self, classifier_type, classifier_configuration, pipeline_configuration, classifier_name, classifier_dir = ""):
        '''
        The super model class. 

        The directories are the base directories for saving the output. 
        Individual directory names will be generated based on the pipeline and
        model configuration. 
        :param classifier_configuration:  classifier specific parameters (e.g. network architecture for Neural Network or ) 
                                dictionary (the loaded config) or a path (provided as str) to the pickle object (object will be loaded)
        :param pipeline_configuration: general configuration used for the selected classifier (e.g. learning rate or selected features)
                                dictionary (the loaded config) or a path (provided as str) to the pickle object (object will be loaded)
        :param classifier_name: a string representing the basename of a classifier --> identifier
        :param pretrained_mode: this mode is on, when a previously generated model is loaded and it is not desired to create output statistics in the standard setting

        '''
        if len(classifier_dir) == 0:
            classifier_dir = CLASSIFIER_DIR

        self.classifier_dir = classifier_dir    # General output dir, in this directory a subdir will be generated 
        self.classifier_output_dir = ""         # The directory where the output specifically for this classifier will be saved

        self.classifier_type = classifier_type

        # If the configuration (pipeline or classification) if provided as path (str) to the pickled object
        # then load from there
        if type(classifier_configuration) == str:
            assert(os.path.exists(classifier_configuration))
            with open(classifier_configuration, 'rb') as f:
                classifier_configuration = pickle.load(f)
        self.classifier_configuration = classifier_configuration 
        if type(pipeline_configuration) == str:
            assert(os.path.exists(pipeline_configuration))
            with open(pipeline_configuration, 'rb') as f:
                pipeline_configuration = pickle.load(f)
        self.pipeline_configuration = pipeline_configuration

        self.train_time = -1
        self.prediction_time = -1

        self.classifier_name = classifier_name

        self.classifier_path = ""           # where model will be saved
        self.prediction_output_path = ""    # csv where the predictions will be saved

        self.trained_classifier_output_dir = ""
        self.predictions_output_dir = ""
        self.evaluation_output_dir = ""

        # if (type(pipeline_configuration) == type(None)) or (type(classifier_configuration) == type(None)):
        #     # TODO pretrained mode so far only for NN case implemented
        #     self.pretrained_mode = True
        #     print("Pipeline or classification pipelines are not delivered. Assuming you are in the mode where a pretrained model has been loaded.")
        #     return

        self.classifier_output_dir = self.get_classifier_output_dir()
        self.set_output_directories()
        if not os.path.exists(self.classifier_output_dir):
            os.makedirs(self.classifier_output_dir)
        # self.create_output_directories()
        self.save_configs()

        self.training_output_specs_file     = ""
        self.prediction_output_specs_file   = ""
        self.evaluation_output_specs_file   = ""

        # Print Configuration to file in the directory
        self.configuration_file = os.path.join(self.classifier_output_dir, "configuration.txt")
        print_configurations_to_file(self.configuration_file, [self.pipeline_configuration, self.classifier_configuration])

    def set_output_directories(self):
        '''
        Setting the relevant output paths.
        '''
        self.trained_classifier_output_dir = self.get_classifier_training_output_dir()
        self.predictions_output_dir = self.get_classifier_predictions_output_dir()
        self.evaluation_output_dir = self.get_classifier_evaluation_output_dir()    

    def set_classifier_output_dir(self, output_dir):
        self.classifier_output_dir = output_dir

        if not os.path.exists(self.classifier_output_dir):
            os.makedirs(self.classifier_output_dir)

    # Setter methods for changing output directories post initialization
    # Move all existing files to the new directory, then delete the old one
    def set_trained_classifier_output_dir(self, output_dir):
        self.trained_classifier_output_dir = output_dir

    def set_predictions_output_dir(self, output_dir):
        self.predictions_output_dir = output_dir

    def set_evaluation_output_dir(self, output_dir):
        self.evaluation_output_dir = output_dir

    # If desired
    def clean_up(self):
        '''
        Remove all directories and files associated with this classifier.
        E.g. use the classifier for making prediction, if the value is below a threshold --> delete
        '''
        shutil.rmtree(self.classifier_output_dir, ignore_errors=False, onerror=None)

    def train(self, train_files):
        '''
        :param train_files: a list of train files
        '''
        self.set_output_directories()
        # Make sure the train files are all in the same parent directory so that they have been generated with the same activity code type dict
        assert(len(list(set([Path(train_f).parent for train_f in train_files]))) == 1)
        # The directory where the data is stored by default has an activity code type dictionary (as pickle object)
        self.activity_code_type_dict = get_dir_activity_code_type_dict(Path(train_files[0]).parent)
        
        self.n_classes = len(list(self.activity_code_type_dict.keys())) # all possible activities        
        self.labels = list(self.activity_code_type_dict.keys())

        if not os.path.exists(self.trained_classifier_output_dir):
            os.makedirs(self.trained_classifier_output_dir)

        save_activity_code_type_dict(self.activity_code_type_dict, self.trained_classifier_output_dir)

        self.train_files = train_files
        self.classifier_path = self.get_trained_classifier_output_path()

        # Print Platform specifications   
        self.training_output_specs_file = self.classifier_path.replace(".h5", "_run_information.txt")
        self.training_output_specs_file = self.training_output_specs_file.replace(".pkl", "_run_information.txt")

    def predict(self, test_file, range = (0, 1)):

        self.set_output_directories()
        if not os.path.exists(self.predictions_output_dir):
            os.makedirs(self.predictions_output_dir)

        self.test_file = test_file
        if self.pipeline_configuration["TRAIN_PER_FILE"]:
            for f in self.classifier_paths:
                if not os.path.exists(f):
                    print(f"Model has not been fully trained. Call the train function again.")
                    sys.exit()
                    return 
        else:
            if not os.path.exists(self.classifier_path):
                if not self.pretrained_mode: 
                    print(f"Model has not been trained yet. Call the train function first on a number of training files or supply a path to the pretrained model in order to be able to make predictions.")
                    sys.exit()

        self.prediction_output_path = self.get_classifier_predictions_output_path(test_file, range = range)
        self.probabilities_output_path = self.get_classifier_probabilities_output_path(test_file, range = range)

        self.prediction_output_specs_file = self.prediction_output_path.replace(".pkl", "_run_information.txt")
        if not os.path.exists(self.prediction_output_specs_file): 
            print_platform_specifications(self.prediction_output_specs_file)

    def evaluate(self, test_file, range = (0, 1.0)):
        self.test_file = test_file
        self.set_output_directories()
        if not os.path.exists(self.evaluation_output_dir):
            os.makedirs(self.evaluation_output_dir)
     
        self.labels = list(self.activity_code_type_dict.keys())

        save_activity_code_type_dict(self.activity_code_type_dict, self.evaluation_output_dir)

        if self.pipeline_configuration["TRAIN_PER_FILE"]: 
            par_dir = str(Path(self.evaluation_output_dir).parent)
        else:
            par_dir = self.evaluation_output_dir
        self.stat_out_file = os.path.join(par_dir, "stat_overview.csv")

        if os.path.exists(self.stat_out_file):
            self.stat_df = pd.read_csv(self.stat_out_file)
        else:
            self.stat_df = pd.DataFrame(columns=["classifier", "test_file", "accuracy", "f1", "loss"])

        self.stat_df_filename = os.path.basename(self.test_file) if not self.iterable_test_file(test_file) else f"{len(test_file)}_files_split_({range[0]},{range[1]})"

        self.stat_df_row = self.stat_df.loc[(self.stat_df["classifier"] == os.path.basename(self.classifier_path)) & (self.stat_df["test_file"] == self.stat_df_filename)]

        return
    
    def set_classifier_path(self, classifier_path):
        '''
        '''
        self.classifier_path = classifier_path


    def print_training_process(self):
        '''
        '''
        original_stdout = sys.stdout
        
        if os.path.exists(self.training_output_specs_file):
            mode = "a"
        else:
            mode = "w"

        with open(self.training_output_specs_file , mode) as f:
            sys.stdout = f # Change the standard output to the file we created.
 
            print(f"Classifier: {self.classifier_path}")
            print(f"Train time: {self.train_time}\n")
            print("Train files")
            for f in self.train_files:
                print(f)
            print("\n")

        sys.stdout = original_stdout
        print("")

    def iterable_test_file(self, test_file):
        '''
        Checks whether the submitted file is an iterable of test_files 
        '''
        if hasattr(test_file, '__iter__'):
            if type(test_file) != str:
                return True
        return False
    
    def print_prediction_process(self):
        '''
        '''
        original_stdout = sys.stdout
        
        if os.path.exists(self.training_output_specs_file):
            mode = "a"
        else:
            mode = "w"

        with open(self.training_output_specs_file , mode) as f:
            sys.stdout = f # Change the standard output to the file we created.

            print(f"Classifier: {self.classifier_path}")
            print(f"Prediction time: {self.prediction_time}\n")
            print(f"Test file: {self.test_file}")

        sys.stdout = original_stdout
        print("")

    def get_classifier_base_dir(self):
        ''' 
        Returns a base directory name as string based on the used configuration.  
        '''
        step_size = self.pipeline_configuration["STEP_SIZE"]
        if self.classifier_type in NN_IMPLEMENTATIONS:
            window_size = self.pipeline_configuration["WINDOW_SIZE"]
            epochs = self.pipeline_configuration["EPOCHS"]
            lr = self.pipeline_configuration["LR"]
            batch_size = self.pipeline_configuration["BATCH_SIZE"]
            base_dir = f"s{step_size}_w{window_size}_e{epochs}_lr{lr}_b{batch_size}"

        else:
            feature_selection_method = self.pipeline_configuration["FEATURE_SELECTION_METHOD"]
            base_dir = f"s_{step_size}_w"
            for w in self.pipeline_configuration["WINDOW_SIZES"]: 
                base_dir += f"_{w}"
            num_features = self.pipeline_configuration["NUM_FEATURES"]
            base_dir += f"_{feature_selection_method}_numf{num_features}"
        
        return base_dir

    def get_pipeline_config_output(self, output_dir):
        return os.path.join(output_dir, "pipeline_config.pkl")
    
    def get_classifier_config_output(self, output_dir):
        return os.path.join(output_dir, "classifier_config.pkl")

    def save_configs(self):
        '''
        Save the classifier and pipeline configuration (python dictionaries) as pickled objects to the model output directory.
        '''
        classifier_config_f = self.get_classifier_config_output(self.classifier_output_dir)
        if not os.path.exists(classifier_config_f): 
            with open(classifier_config_f, 'wb') as f:
                pickle.dump(self.classifier_configuration, f, protocol=pickle.HIGHEST_PROTOCOL)
        pipeline_config_f = self.get_pipeline_config_output(self.classifier_output_dir) 
        if not os.path.exists(pipeline_config_f):
            with open(pipeline_config_f, 'wb') as f:
                pickle.dump(self.pipeline_configuration, f, protocol=pickle.HIGHEST_PROTOCOL)

    def get_ground_truth_prediction_values(self, test_file, gt_file = None, range = (0, 1.0), label_mapping = None):
        '''
        Compute accuracy and f1 score and create confusion matrices.
        '''
        if self.iterable_test_file(test_file):

            if type(gt_file) != type(None):
                assert(self.iterable_test_file(gt_file))
            else: 
                gt_file = test_file

            gt_values = []
            pred_values = []
            probabilities_df = pd.DataFrame()
            for t, g in zip(test_file, gt_file):

                gt_values_f, pred_values_f, probabilities_df_f = self.get_ground_truth_prediction_values(t, g, range = range, label_mapping = label_mapping)

                gt_values.extend(gt_values_f)
                pred_values.extend(pred_values_f)
                probabilities_df = pd.concat([probabilities_df, probabilities_df_f], ignore_index=True)

            return gt_values, pred_values, probabilities_df

        self.prediction_output_path = self.get_classifier_predictions_output_path(test_file, range = range)
        self.probabilities_output_path = self.get_classifier_probabilities_output_path(test_file, range = range)

        if not os.path.exists(self.prediction_output_path):
            self.predict(test_file, range = range)
            
        if type(gt_file) == type(None):
            gt_file = test_file
            assert(os.path.exists(test_file))

        self.predictions = self.get_predictions_from_file(self.prediction_output_path)
        self.probabilities = self.get_probabilities_from_file(self.probabilities_output_path)
        self.gt = self.get_gt_from_file(gt_file)

        if type(label_mapping) != type(None):
            # Map the predictions
            self.predictions = [label_mapping[p] for p in self.predictions]

        # merge ground truth and prediction only where the index exists in both
        merged_df = pd.concat([self.gt, self.predictions], axis = 1, keys = [LABEL_COL, PREDICTION_COL], join = "inner")

        probability_cols = self.probabilities.columns.values
        merged_df = pd.concat([merged_df, self.probabilities], axis = 1, join = "inner")

        gt_values = merged_df[LABEL_COL].values
        prediction_values = merged_df[PREDICTION_COL].values
        probabilities = merged_df[probability_cols]

        return gt_values, prediction_values, probabilities

    def evaluate_from_prediction_gt_prob_values(self, gt_values, prediction_values, probability_df):
        '''
        :param gt_values: array or list of same length as predictions                       ground truth activitiy class
        :param prediction_values: array or list of same length as gt_values                 predicted activity class
        :param probability_df: dataframe of same length, column names indicate the class    predicted probability per activity class
        '''
        
        self.accuracy, self.f1_score, cm, cm_normalized, display_labels = get_evaluation_metrics(gt_values, prediction_values, activity_code_type_dict = self.activity_code_type_dict, labels = self.labels)

        # create the confusion matrices and save them
        cm_title =  f"Confusion matrix for {self.classifier_name}\naccuracy = {self.accuracy}\nf1_score = {self.f1_score}"
        out_save_fname = os.path.join(self.evaluation_output_dir, f"{self.classifier_name}_cm.png")
        visualize_confusion_matrix(cm, display_labels = display_labels, title = cm_title, out_save_fname = out_save_fname)
        out_save_fname_normalized = out_save_fname.replace(".png", "_normalized.png")
        visualize_confusion_matrix(cm_normalized, display_labels = display_labels, title = cm_title, out_save_fname = out_save_fname_normalized)

        label_distr_out_save_fname = os.path.join(self.evaluation_output_dir, f"{self.classifier_name}_label_distribution.png")
        label_distribution = get_label_distr_dict(self.activity_code_type_dict, gt_values)
        self.save_label_distribution(label_distribution, label_distr_out_save_fname, dataset_type = "evaluation")

        self.cross_entropy_loss = get_cross_entropy_loss(gt_values, probability_df)

        return self.accuracy, self.f1_score, self.cross_entropy_loss

    def evaluation_to_statistic_file(self):
        '''
        Prints the evaluation to a csv file for fast overview of the results
        '''
        if len(self.stat_df_row) == 0:
            self.stat_df = self.stat_df.append({"classifier": os.path.basename(self.classifier_path), "test_file": self.stat_df_filename, "accuracy": self.accuracy, "f1": self.f1_score, "ce-loss": self.cross_entropy_loss}, ignore_index = True)
            self.stat_df.to_csv(self.stat_out_file, index=False)

    def get_classifier_output_dir(self):
        '''
        Get the specific output directory for this classifier, based on the configuration. Check if a directory already exists with the same name.
        If so, check if the configurations saved in the directory are the same as self.pipeline_config and self.classifier_config 
            yes --> this is the output directory
            no  --> add a C{index} str at the end, index will increas  
        '''
        if self.pretrained_mode:
            return self.classifier_dir

        classifier_base_output_dir = os.path.join(self.classifier_dir, f"{self.classifier_type}/{self.get_classifier_base_dir()}")

        max_dirs = 100

        c = -1
        while c < max_dirs:
            c+=1
            classifier_output_dir = classifier_base_output_dir + f"_C{c}"

            if not os.path.exists(classifier_output_dir):
                return classifier_output_dir
            
            else:
                existing_classifier_config_file = self.get_classifier_config_output(classifier_output_dir)
                exisiting_pipeline_config_file = self.get_pipeline_config_output(classifier_output_dir)
                if  os.path.exists(existing_classifier_config_file) & os.path.exists(exisiting_pipeline_config_file):
                    with open(exisiting_pipeline_config_file, 'rb') as f:
                        existing_pipeline_config = pickle.load(f)
                    with open(existing_classifier_config_file, 'rb') as f:
                        existing_classifier_config = pickle.load(f)

                    if (existing_classifier_config == self.classifier_configuration):
                        pipeline_configuration_copy = copy.copy(self.pipeline_configuration)
                        # del pipeline_configuration["STEP_SIZE"]
                        del pipeline_configuration_copy["STEP_SIZE_TEST"]
                        # del exisiting_pipeline_config["STEP_SIZE"]
                        del existing_pipeline_config["STEP_SIZE_TEST"]

                        if (existing_pipeline_config == pipeline_configuration_copy):
                            return classifier_output_dir
                        else:
                            continue
                    else:
                        continue
                else:
                    return classifier_output_dir
        return None

    def get_classifier_output_subdir(self, mode, step_size):
        '''
        :param mode: either Training, Prediction, Evaluation
        '''
        if self.pipeline_configuration["TRAIN_PER_FILE"]:
            # If train per file is true --> create another subdirectory
            train_per_file_subdir = f"/{self.classifier_name}"
        else:
            train_per_file_subdir = ""
        output_dir = self.classifier_output_dir + f"/{mode}_{step_size}{train_per_file_subdir}"
        return output_dir

    def get_classifier_training_output_dir(self):
        '''
        Return the directory 
        '''
        step_size_train = self.pipeline_configuration["STEP_SIZE"]
        return self.get_classifier_output_subdir("Training", step_size_train)

    def get_classifier_predictions_output_dir(self):
        '''
        '''
        step_size_test = self.pipeline_configuration["STEP_SIZE_TEST"]
        return self.get_classifier_output_subdir("Prediction", step_size_test)

    def get_classifier_evaluation_output_dir(self):
        '''
        '''
        step_size_test = self.pipeline_configuration["STEP_SIZE_TEST"]
        return self.get_classifier_output_subdir("Evaluation", step_size_test)

    def get_trained_classifier_output_path(self):
        if self.classifier_type in NN_IMPLEMENTATIONS:
            model_output_path = os.path.join(self.trained_classifier_output_dir, self.classifier_name + ".h5")
        else:
            model_output_path = os.path.join(self.trained_classifier_output_dir, self.classifier_name + ".pkl")
        return model_output_path
    
    def get_classifier_predictions_output_path(self, test_file, range = (0, 1)):
        test_file_basename = os.path.basename(test_file).split(".")[0]

        if range != (0, 1.0): 
            test_file_basename += (f"_({range[0]},{range[1]})")

        predictions_output_path = os.path.join(self.predictions_output_dir, f"{self.classifier_name}_on_{test_file_basename}_predictions.pkl")

        return predictions_output_path
    
    def get_classifier_probabilities_output_path(self, test_file, range = (0,1)):
        '''
        '''
        pred_output_path = self.get_classifier_predictions_output_path(test_file, range = range)
        probabilities_path = pred_output_path.replace("predictions", "probabilities") 
        return probabilities_path

    def set_step_size_test(self, step_size):
        '''
        '''
        self.pipeline_configuration["STEP_SIZE_TEST"] = step_size

    def visualize_prediction_vs_gt(self,
                        predictions_file,
                        gt_file,
                        label_mapping = None,
                        signals = ["JOINT_ANGLE", "JOINT_LOAD", LABEL_COL, PREDICTION_COL, "RI_RULID1"],
                        data_scaling_factors = [1, 1/100, 1/10, 1/10, 1]):
        '''
        Print
        '''
        self.predictions = load_from_pickle(predictions_file)
        if type(self.predictions) == type(pd.DataFrame()):
            if INDEX_COL in self.predictions.columns.values:
                self.predictions.set_index(INDEX_COL)  
            self.predictions = self.predictions[PREDICTION_COL]
        # pd.read_csv(predictions_file, index_col=INDEX_COL, usecols=[INDEX_COL, PREDICTION_COL])

        if type(label_mapping) != type(None):
            # Map the predictions
            self.predictions = apply_label_mapping(self.predictions, label_mapping)

        if gt_file.endswith(".csv"):
            ground_truth_df = pd.read_csv(gt_file, index_col=INDEX_COL)
        else:
            ground_truth_df = load_from_pickle(gt_file)
            if INDEX_COL in ground_truth_df.columns.values:
                ground_truth_df.set_index(INDEX_COL)

        df = ground_truth_df.join(self.predictions).dropna()        
        matplotlib.use("TkAgg")

        try:
            plot_signals_from_df(df, signals, data_scaling_factors = data_scaling_factors, title = f"{self.classifier_type} on {self.classifier_name}\naccuracy = {self.accuracy}, f1 = {self.f1_score}", linestyles = ["-","-","-","--", "-"], x = df.index.values)
        except:
            # Leipzig data does not have rul id 
            plot_signals_from_df(df, signals[:-1], data_scaling_factors = data_scaling_factors[:-1], title = f"{self.classifier_type} on {self.classifier_name}\naccuracy = {self.accuracy}, f1 = {self.f1_score}", linestyles = ["-","-","-","--"][:-1], x = df.index.values)

        # matplotlib.use("Agg")

    def get_predictions_from_file(self, prediction_output_path):
        '''
        Load the predictions and ground truth from the file generated after calling self.predict()

        :return: Series with prediction values and index
        '''
        print(f"Loading predictions from {prediction_output_path}")

        predictions = load_from_pickle(prediction_output_path) # predictions is a pandas series with the index

        if type(predictions) == type(pd.DataFrame()):
            if INDEX_COL in predictions.columns.values:
                predictions.set_index(INDEX_COL)
            
            predictions = predictions[PREDICTION_COL]

        return predictions

    def get_probabilities_from_file(self, probabilities_output_path):

        probabilities = load_from_pickle(probabilities_output_path)

        if INDEX_COL in probabilities.columns.values:
            probabilities =probabilities.set_index(INDEX_COL)

        return probabilities

    def get_gt_from_file(self, gt_file):
        '''
        '''
        print(f"Loading ground truth from {gt_file}")     
        gt_df = load_from_pickle(gt_file)

        if INDEX_COL in gt_df.columns.values:
            gt_df.set_index(INDEX_COL)
        
        gt = gt_df[LABEL_COL]
        
        return gt


    def get_pretrained_mode(self):
        '''
        Determine whether pretrained mode is on
        If on --> load the model with initialization
        '''
        pretrained_mode = False
        if type(self.pretrained_path) != type(None): 
            if os.path.exists(self.pretrained_path):
                pretrained_mode = True

        return pretrained_mode

    def save_label_distribution(self, label_distribution, classifier_path, dataset_type = "training"): 
        '''
        Save the label distribution visualised as pie chart. 
        '''
        num_samples = sum(list(label_distribution.values()))
        labels = [f"{self.activity_code_type_dict[x]}" for x in label_distribution.keys()]

        # Classifier names either are pkl or h5
        output_file = classifier_path.replace(".h5", ".png")
        output_file = output_file.replace(".pkl", ".png")
        output_file = output_file.replace(".png", f"_{dataset_type}_dataset_overview.png")

        visualize_dict_as_pie_chart(label_distribution, output_file=output_file, labels = labels,title=f"accumulated activity overview ({dataset_type})\ntotal of {num_samples} samples")
        plt.close()
        return

class NeuralNetwork(Classifier):

    def __init__(self, classifier_type, classifier_configuration, pipeline_configuration, classifier_name = "", classifier_output_dir = ""):
        
        self.pretrained_path = pipeline_configuration["PRETRAINED"]
        self.pretrained_mode = self.get_pretrained_mode()

        super().__init__(classifier_type, classifier_configuration, pipeline_configuration, classifier_name, classifier_output_dir)
        
        if self.pretrained_mode:
            self.load_classifier_from_path(path = self.pretrained_path)
    
    def set_train_files(self, train_files):
        self.train_files = train_files

    def set_test_file(self, test_file):
        self.test_file = test_file

    def get_train_files(self):
        return self.train_files
    
    def get_test_file(self):
        return self.test_file

    def load_classifier_from_path(self, path):
        '''
        '''
        assert os.path.exists(path)

        print(f"Loading model from {path}")
        # TODO save during training and load custom optimizer 
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.pipeline_configuration["LR"])
        lr_metric = get_lr_metric(optimizer)
        self.model = tf.keras.models.load_model(path, custom_objects={"f1": f1, "ce_loss": ce_loss}) 
        self.model.run_eagerly = True #"lr": lr_metric, 
        self.classifier_path = path

    def save_dataset_distribution(self, dataset, dataset_type = "train", output_base_path = ""):
        '''
        Save the label distribution of the train dataset.
        :param dataset_type: any string, should be either train validation or test
        '''
        if len(output_base_path) == 0:
            output_base_path = self.classifier_path

        label_distribution = get_label_distribution_from_dataset(dataset, activity_code_type_dict=self.activity_code_type_dict)
        super().save_label_distribution(label_distribution, output_base_path, dataset_type = dataset_type)

    def get_best_history(self, history):
        '''
        :param history: a trained keras model's .history.history element.
        '''
        loss_values = history["val_loss"] if "val_loss" in history.keys() else history["loss"]
        best_index = loss_values.index(min(loss_values))

        history_out = {}
        for k, history_values in history.items():
            history_out[k] = history_values[best_index]

        return history_out

    def train(self, train_files):
        self.set_output_directories()
        super().train(train_files)

        if os.path.exists(self.classifier_path):
            if self.is_history_saved():
                self.load_classifier_from_path(self.classifier_path)
                history = self.load_history() 
                return self.get_best_history(history)

        if os.path.exists(self.training_output_specs_file): 
            os.remove(self.training_output_specs_file)
        print_platform_specifications(self.training_output_specs_file)

        signal_cols     = self.pipeline_configuration["SIGNALS"]
        window_size     = self.pipeline_configuration["WINDOW_SIZE"]
        step_size       = self.pipeline_configuration["STEP_SIZE"]
        batch_size      = self.pipeline_configuration["BATCH_SIZE"]
        early_stopping  = self.pipeline_configuration["EARLY_STOPPING"]
        epochs          = self.pipeline_configuration["EPOCHS"]
        train_val_split = self.pipeline_configuration["TRAIN_VAL_SPLIT"]
        sampling_mode   = self.pipeline_configuration["SAMPLING_MODE"]

        # Determine input shape
        self.input_shape = (window_size, len(signal_cols))

        # Get the indices
        dataset_generator_args = [train_files, signal_cols, LABEL_COL, window_size, step_size, True, batch_size, list(self.activity_code_type_dict.keys()), list(self.activity_code_type_dict.values()), (0, 1), sampling_mode]
        
        if train_val_split == 1.0:
            loc_indices_list, loc_indices_files_list = get_loc_indices_list_from_files(train_files, step_size = step_size, window_size=window_size, sampling_mode=sampling_mode, dataset_range = (0, 1), feature_file = False)
            dataset_generator_args.append(loc_indices_files_list)

            self.train_dataset = tf.data.Dataset.from_generator(tf_dataset_generator, args = dataset_generator_args, output_types = (tf.float32, tf.float32),
                                                output_shapes = ((None, window_size, len(signal_cols)),(None, self.n_classes)))
            self.val_dataset = None
            use_val_data = False
        else:
            train_dataset_args = list(np.array(dataset_generator_args, dtype = "object"))
            val_dataset_args = list(np.array(dataset_generator_args, dtype = "object"))

            train_loc_indices_list, train_loc_indices_files_list = get_loc_indices_list_from_files(train_files, step_size = step_size, window_size=window_size, sampling_mode=sampling_mode, dataset_range = (0, train_val_split), feature_file = False, use_total_label_distribution=True)
            train_dataset_args.append(train_loc_indices_files_list)
            val_loc_indices_list, val_loc_indices_files_list = get_loc_indices_list_from_files(train_files, step_size = step_size, window_size=window_size, sampling_mode=sampling_mode, dataset_range = (train_val_split, 1.0), feature_file = False, use_total_label_distribution=True)
            val_dataset_args.append(val_loc_indices_files_list)

            self.train_dataset = tf.data.Dataset.from_generator(tf_dataset_generator, args = train_dataset_args, output_types = (tf.float32, tf.float32),
                                                output_shapes = ((None, window_size, len(signal_cols)),(None, self.n_classes)))

            self.val_dataset = tf.data.Dataset.from_generator(tf_dataset_generator, args = val_dataset_args, output_types = (tf.float32, tf.float32),
                                    output_shapes = ((None, window_size, len(signal_cols)),(None, self.n_classes)))
            use_val_data = True

        self.save_dataset_distribution(self.train_dataset)
        if type(self.val_dataset) != type(None):
            self.save_dataset_distribution(self.val_dataset, dataset_type = "validation")

        model = get_compiled_model(self.input_shape, self.n_classes, self.classifier_type, self.pipeline_configuration, network_configuration = self.classifier_configuration, use_val_data=use_val_data)

        model_summary_file = os.path.join(self.classifier_output_dir, f"{self.classifier_type}_modelsummary.txt")
        with open(model_summary_file, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        # Start the training and monitor training time
        
        original_stdout = sys.stdout
        train_monitor_output_txt = self.trained_classifier_output_dir + "/train_monitoring.txt"

        print(f"Starting training. Saving output to {train_monitor_output_txt}")

        with open(train_monitor_output_txt , "w") as f:
            # history = model.fit(self.train_dataset, epochs = 10)
            
            sys.stdout = f # Change the standard output to the file we created.
    
            train_start_time = process_time()
            classifier_path_training_tmp = self.classifier_path.replace(".h5", "_training_tmp.h5")
            self.model = train_model(model, self.train_dataset, epochs, self.pipeline_configuration, classifier_path_training_tmp, self.activity_code_type_dict, early_stopping_acc_threshold = early_stopping, val_set=self.val_dataset)

            try:
                os.rename(classifier_path_training_tmp, self.classifier_path) # rename the file after trainiing
            except: 
                model.save(self.classifier_path)
        sys.stdout = original_stdout

        train_stop_time = process_time()
        self.train_time = train_stop_time - train_start_time
        
                
        print(f"Trained model in {self.train_time}.")

        # self.model.save(self.classifier_output_path)
        print(f"Model saved under {self.classifier_path}.")

        self.print_training_process()
        history = self.model.history.history
        self.save_history(history)

        return self.get_best_history(history)

    def get_history_path(self, classifier_path):
        '''
        '''
        return classifier_path.replace(".h5", "_history.pkl")
    
    def get_history(self):
        '''
        '''
        history_path = self.get_history_path(self.classifier_path)
        assert(os.path.exists(history_path))
        history = load_from_pickle(history_path)
        return history

    def save_history(self, history):
        '''
        Plot the loss and the f1 score as plot. 
        '''
        
        save_history_plots(history, self.classifier_path)

        save_as_pickle(history, self.get_history_path(self.classifier_path))
        
        return
    
    def load_history(self):
        '''
        Load the model's history
        '''
        return load_from_pickle(self.get_history_path(self.classifier_path))

    def is_history_saved(self):
        '''
        Check if the history has been saved --> indicator that training was finished properly
        '''
        dir = os.path.dirname(self.classifier_path)
        basename = os.path.basename(self.classifier_path).replace(".h5", "")

        history_paths = glob.glob(dir + f"/{basename}*") # paths to plots, check if these exist

        if len(history_paths) > 2:
            return True
        else:
            return False

    def predict(self, test_file, label_mapping = None, range = (0, 1)):
        ''' 
        Use the trained model saved in self.model to predict the submitted test file. Results will be saved in
        a output csv in the self.predictions_output_dir folder. 
        '''
        super().predict(test_file)

        # self.activity_code_type_dict = get_dir_activity_code_type_dict("C:/Users/fichtel/Ottobock SE & Co. KGaA/Deep Learning - General/AI-Pipeline/aos/02-Activities-GT")
        if self.pretrained_mode:
            self.n_classes = len(self.activity_code_type_dict)

        signal_cols     = self.pipeline_configuration["SIGNALS"]
        window_size     = self.pipeline_configuration["WINDOW_SIZE"]
        batch_size      = self.pipeline_configuration["BATCH_SIZE"]
        step_size       = self.pipeline_configuration["STEP_SIZE_TEST"]
        sampling_mode   = self.pipeline_configuration["SAMPLING_MODE"]
        if self.pipeline_configuration["STEP_SIZE_TEST"] == 1:
            sampling_mode = "fixed"

        loc_indices_list, loc_indices_files_list = get_loc_indices_list_from_files([test_file], step_size, window_size, sampling_mode, dataset_range = range)

        args = [[test_file], signal_cols, LABEL_COL, window_size, step_size, False,  batch_size, list(self.activity_code_type_dict.keys()), list(self.activity_code_type_dict.values()), range, sampling_mode, loc_indices_files_list]

        self.test_dataset = tf.data.Dataset.from_generator(tf_dataset_generator, args=args, output_types = (tf.float32, tf.float32),
                                            output_shapes = ((None, window_size, len(signal_cols)),(None, self.n_classes)))

        self.save_dataset_distribution(self.test_dataset, dataset_type = "test", output_base_path = self.prediction_output_path)

        if os.path.exists(self.prediction_output_path):
            self.predictions = super().get_predictions_from_file(self.prediction_output_path)
        else:
            prediction_start_time = process_time()
            
            print(f"{self.classifier_type} predicting on {test_file}")
            self.predictions, self.probabilities = get_prediction_nn(self.model, self.test_dataset, activity_dict = self.activity_code_type_dict)
            
            prediction_stop_time = process_time()
            self.prediction_time = prediction_stop_time - prediction_start_time
            print(f"Prediction done in {self.prediction_time}.")
            self.predictions = pd.Series(self.predictions, index = loc_indices_list[0], name = PREDICTION_COL)
            self.probabilities = pd.DataFrame(self.probabilities, index = loc_indices_list[0], columns=list(self.activity_code_type_dict.keys()))

            save_as_pickle(self.predictions, self.prediction_output_path)
            save_as_pickle(self.probabilities, self.probabilities_output_path)
            print(f"Saved prediction under in {self.prediction_output_path}.")
        
        if type(label_mapping) != type(None):
            self.predictions = apply_label_mapping(self.predictions, label_mapping)
        
        self.probabilities = self.get_probabilities_from_file(self.probabilities_output_path)

        return self.predictions, self.probabilities

    def predict_dataset(self, test_dataset, label_mapping = None):
        '''
        :param dataset: predict a windowed generator dataset
        '''
        predictions = get_prediction_nn(self.model, test_dataset, activity_dict = self.activity_code_type_dict)

        if type(label_mapping) != type(None):
            
            predictions = apply_label_mapping(predictions, label_mapping)

        return np.array(predictions)
    
    def predict_dataset_probabilities(self, test_dataset, label_mapping = None):

        probabilities = get_probabilities_nn(self.model, test_dataset)

        encoder = get_one_hot_encoder(self.activity_code_type_dict)

        probabilities_df = pd.DataFrame(data = probabilities, 
                columns = encoder.categories_[0])

        if type(label_mapping) != type(None):
            label_mapping_df = label_mapping_dict_to_df(label_mapping)
            probabilities_df_out = pd.DataFrame()

            for out_label in list(set(list(label_mapping_df["label_out"].values))):
                
                labels_in = label_mapping_df.loc[label_mapping_df["label_out"] == out_label]["label_in"].values

                probabilities_df_out[out_label] = probabilities_df[labels_in].sum(axis = "columns")

            probabilities_df = probabilities_df_out
        
        return probabilities_df

    def evaluate(self, test_file, label_mapping = None):
        '''
        '''
        super().evaluate(test_file)

        print(f"Running nn evaluation for test file {self.test_file}. Results will be saved in {self.evaluation_output_dir}.")
        
        gt_values, prediction_values, probabilities_df =  super().get_ground_truth_prediction_values(test_file, label_mapping = label_mapping)

        accuracy, f1_score, cross_entropy_loss = super().evaluate_from_prediction_gt_prob_values(gt_values, prediction_values, probabilities_df)

        super().evaluation_to_statistic_file()

        return accuracy, f1_score, cross_entropy_loss 

    def visualize_prediction_vs_gt(self, test_file, label_mapping = None):
        '''
        In the NN case gt_file == test_file
        so far this only is possible for the case STEP_SIZE_TEST == 1
        # TODO
        '''
        self.predict(test_file)
        super().visualize_prediction_vs_gt(self.prediction_output_path, self.test_file, label_mapping = label_mapping)

class ShallowClassifier(Classifier):

    def __init__(self, type, classifier_configuration, pipeline_configuration, classifier_name = "", classifier_output_dir = CLASSIFIER_DIR):
        
        self.load_pretrained_options_shallow(pipeline_configuration)
        self.pretrained_mode = self.get_pretrained_mode_shallow() 

        super().__init__(type, classifier_configuration, pipeline_configuration, classifier_name, classifier_output_dir)

        if self.pipeline_configuration["TRAIN_PER_FILE"]:
            self.classifier_path += "_train_per_file"
        self.classifier_paths = [] # list of classifier paths in the case of train per file 
        self.classifier_dataset_size_dict = {} # to save number of training samples, can be used for ensemble learning to give classifiers importance based on the train data size

        if self.pretrained_mode:
            self.load_classifier_from_path(path = self.pretrained_path)
            self.load_transformer_from_path(path = self.transformer_path)
            self.load_selected_features_from_path(path = self.selected_feature_cols_path)

    def load_pretrained_options_shallow(self, pipeline_configuration):
        '''
        Check if it's desired to load a pretrained model. If not the respective
        class variables will be done and via --> self.get_pretrained_mode_shallow
        will be assigned False
        '''
        if "PRETRAINED" in pipeline_configuration.keys(): 
            self.pretrained_path = pipeline_configuration["PRETRAINED"]
        else:
            self.pretrained_path = None
        if "TRANSFORMER" in pipeline_configuration.keys(): 
            self.transformer_path = pipeline_configuration["TRANSFORMER"]
        else:
            self.transformer_path = None
        if "FEATURE_COLS" in pipeline_configuration.keys():
            self.selected_feature_cols_path = pipeline_configuration["FEATURE_COLS"]
        else:
            self.selected_feature_cols_path = None 
        return


    def get_pretrained_mode_shallow(self):
        '''
        '''
        pretrained_mode = False
        if self.get_pretrained_mode():
            if os.path.exists(self.transformer_path):
                if os.path.exists(self.selected_feature_cols_path):
                    pretrained_mode = True

        return pretrained_mode

    def get_transformer_path(self):
        '''
        PCA feature transformer
        '''
        n_features = self.pipeline_configuration["NUM_FEATURES"]
        return self.classifier_path.replace(".pkl", f"_transformer_{n_features}.pkl")
    
    def get_selected_features_path(self):
        '''
        Get selected features list
        '''
        n_features = self.pipeline_configuration["NUM_FEATURES"]
        return self.classifier_path.replace(".pkl", f"_selected_features_{n_features}.pkl")

    def get_classifier_dataset_size(self):
        '''
        When using "TRAIN_PER_FILE", i.e. ensemble learning method, 
        save the dataset size. This then later can be used to 
        determine a classifier importance when making predictions.
        '''
        return self.classifier_path.replace(".pkl", f"_dataset_sizes_per_classifier_dict.pkl")


    def get_selected_features_and_transformer(self, train_files, loc_indices_list = None):
        '''
        :return:    feature_transformer --> for dimensionality reduction using pca. Will be None if mutual information is used.
                    selected_feature_cols --> list of selected features
        '''
        transformer_path = self.get_transformer_path()
        selected_feature_path = self.get_selected_features_path()

        feature_selection_method = self.pipeline_configuration["FEATURE_SELECTION_METHOD"]

        if os.path.exists(transformer_path) & os.path.exists(selected_feature_path):
            feature_transformer = load_from_pickle(transformer_path)
            selected_feature_cols = load_from_pickle(selected_feature_path)
        elif os.path.exists(selected_feature_path) & (feature_selection_method == "mutual_info"):
            feature_transformer = None
            selected_feature_cols = load_from_pickle(selected_feature_path)
        else:
            print("Performing feature selection using train files.")

            num_features = self.pipeline_configuration["NUM_FEATURES"]
            selection_per_file = self.pipeline_configuration["FEATURE_SELECTION_PER_FILE"]

            selected_feature_cols, feature_transformer = feature_selection(train_files, feature_selection_method, num_features, selection_per_file, loc_indices_list)
            if feature_selection_method == "pca":
                save_as_pickle(feature_transformer, transformer_path)
            save_as_pickle(selected_feature_cols, selected_feature_path)
            
        return selected_feature_cols, feature_transformer

    def load_classifier_from_path(self, path):
        '''
        Load a shallow classifier from path. 
        :param path: #TODO implement if list --> assigne self.classifier_paths
        So far only for "TRAIN_PER_FILE" = False
        '''
        if type(path) == str:
            if not os.path.exists(path):
                print(f"Model does not yet exist: {path}\nTrain first or load a valid path.")
                return
            self.classifier_path = path
        
        if type(path) == list:
            for p in path:
                if not os.path.exists(path): 
                    print(f"Model does not yet exists:{p}")
                    return 
            self.classifier_paths = path

        with open(path, 'rb') as model_f:
            self.model = pkl.load(model_f)

    def load_transformer_from_path(self, path):
        if not os.path.exists(path):
            print(f"Transformer does not yet exist: {path}")
        
        with open(path, 'rb') as transformer_f:
            self.feature_transformer = pkl.load(transformer_f)
    
    def load_selected_features_from_path(self, path): 
        if not os.path.exists(path):
            print(f"Selected features does not yet exist: {path}")
        
        with open(path, 'rb') as selected_features_f:
            self.selected_feature_cols = pkl.load(selected_features_f)

    def save_dataset_distribution_from_labels(self, labels, classifier_path, dataset_type = "train"):
        '''
        Save the label distribution of the train dataset.
        :param dataset_type: any string, should be either train validation or test
        '''
        label_distribution = get_label_distr_dict(self.activity_code_type_dict, labels)
        super().save_label_distribution(label_distribution, classifier_path, dataset_type = dataset_type)

    def train(self, train_files):
        '''
        '''
        super().train(train_files)

        history_path = self.get_history_path()

        # TODO for TRAIN_PER_FILE --> enable loading history 
        if not self.pipeline_configuration["TRAIN_PER_FILE"]:
            if os.path.exists(self.classifier_path) & os.path.exists(self.get_transformer_path()) & os.path.exists(self.get_selected_features_path()) & os.path.exists(history_path):
                # with open(self.classifier_path, 'rb') as model_f:
                # loads the model in self.model
                self.load_classifier_from_path(self.classifier_path)
                self.load_transformer_from_path(self.get_transformer_path())
                self.load_selected_features_from_path(self.get_selected_features_path())
                history = load_from_pickle(history_path)                
                return history
            
        else: 
            # history = load_from_pickle
            print("") # TODO implement
            # return None

        step_size = self.pipeline_configuration["STEP_SIZE"]
        sampling_mode = self.pipeline_configuration["SAMPLING_MODE"]
        train_val_split =  self.pipeline_configuration["TRAIN_VAL_SPLIT"]
        window_size = np.max(self.pipeline_configuration["WINDOW_SIZES"])

        loc_indices_list, loc_indices_files_list = get_loc_indices_list_from_files(train_files, step_size = step_size, window_size=window_size, sampling_mode=sampling_mode, dataset_range = (0, train_val_split), feature_file = True, use_total_label_distribution=True)

        self.selected_feature_cols, self.feature_transformer = self.get_selected_features_and_transformer(self.train_files, loc_indices_list)
        
        if self.pipeline_configuration["TRAIN_PER_FILE"]:
            self.classifier_dataset_size_f = self.get_classifier_dataset_size()

            if os.path.exists(self.classifier_dataset_size_f):
                self.classifier_dataset_size_dict = load_from_pickle(self.classifier_dataset_size_f)

            for train_file, loc_indices in zip(train_files, loc_indices_list):
                train_file_name = os.path.basename(train_file).replace(".pkl", "")
                c_path = self.classifier_path.replace(".pkl", f"_trained_on_{train_file_name}.pkl")
                if train_val_split != 1.0:
                    c_path = c_path.replace(".pkl", f"_{train_val_split}.pkl")

                if not os.path.exists(c_path):
                    X_train, Y_train = get_data_labels_from_file(train_file, selected_feature_cols = self.selected_feature_cols, is_training = True, step_size = step_size, data_transformer = self.feature_transformer, loc_indices= loc_indices)
                    self.save_dataset_distribution_from_labels(Y_train, c_path, dataset_type = "train")
                    model = fit_classifier(X_train, Y_train, self.classifier_type, c_path, output_specs_file = self.training_output_specs_file, classifier_config = self.classifier_configuration)
                    self.classifier_dataset_size_dict[c_path] = len(Y_train)
                self.classifier_paths.append(c_path)

            save_as_pickle(self.classifier_dataset_size_dict, self.classifier_dataset_size_f)
            
        else:
            X_train, Y_train = get_data_labels_from_files(train_files, selected_feature_cols = self.selected_feature_cols, is_training = True, step_size = self.pipeline_configuration["STEP_SIZE"], transformer = self.feature_transformer, loc_indices_list = loc_indices_list)
            self.save_dataset_distribution_from_labels(Y_train, self.classifier_path, dataset_type = "train")
            self.model = fit_classifier(X_train, Y_train, self.classifier_type, self.classifier_path, output_specs_file = self.training_output_specs_file, classifier_config = self.classifier_configuration)
            classifier_history_path = self.classifier_path

        self.print_training_process()
        
        # Create the loc indices files
        loc_indices_list, loc_indices_files_list = get_loc_indices_list_from_files(train_files, step_size = step_size, window_size=window_size, sampling_mode=sampling_mode, dataset_range = (train_val_split, 1.0), feature_file = True, use_total_label_distribution=True)

        # Evaluate
        step_size_test = self.pipeline_configuration["STEP_SIZE_TEST"]
        self.pipeline_configuration["STEP_SIZE_TEST"] = self.pipeline_configuration["STEP_SIZE"] # temporarily change step size
        history = {}
        train_acc, train_f1, ce_loss = self.evaluate(train_files, range = (0, train_val_split))
        history["accuracy"] = train_acc
        history["f1"] = train_f1
        history["loss"] = ce_loss
        if train_val_split != 1.0: 
            val_acc, val_f1, val_ce_loss = self.evaluate(train_files, range = (train_val_split, 1.0)) 
            history["val_accuracy"] = val_acc
            history["val_f1"] = val_f1
            history["val_loss"] = val_ce_loss
        self.pipeline_configuration["STEP_SIZE_TEST"] = step_size_test 

        self.save_history(history, history_path)

        return history

    def get_history_path(self):
        '''
        '''
        if not self.pipeline_configuration["TRAIN_PER_FILE"]:
            history_path = self.classifier_path.replace(".pkl", "_history.pkl")
        else:
            history_path = self.trained_classifier_output_dir + f"{self.classifier_name}_history.pkl"
        return history_path
    
    def save_history(self, history, history_path):
        '''
        Plot the loss and the f1 score as plot. 
        '''
        save_as_pickle(history, history_path)
        return

    def predict_df_from_loc_indices(self, feature_df, loc_indices): 
        '''
        '''
        X_test, Y_test = get_data_labels_from_df(feature_df, selected_feature_cols = self.selected_feature_cols, is_training = False, data_transformer = self.feature_transformer, loc_indices=loc_indices)

        if self.pipeline_configuration["TRAIN_PER_FILE"]:     # TODO implement TRAIN_PER_FILE case #TODO save 
            probabilities = []
            for classifier_filepath in self.classifier_paths:
                model = load_from_pickle(classifier_filepath)
                Y_pred_prob = predict_probability(model, X_test)
                Y_pred_prob.index = loc_indices
                if INDEX_COL not in Y_pred_prob.columns.values:
                    Y_pred_prob[INDEX_COL] = Y_pred_prob.index
                probabilities.append(Y_pred_prob)

        else: 
            model = load_from_pickle(self.classifier_path)
            probabilities = predict_probability(model, X_test)
            probabilities.index = loc_indices

            predictions = probabilities_df_to_classification(probabilities)
            
        return predictions, probabilities

    def predict(self, test_file, range = (0, 1.0)):
        '''
        :param test_file: either one test file as string! or an iterable of test files ["test_file1.pkl", "test_file2.pkl", "test_file3.pkl"....]
        '''
        # check if the test_file is iterable and not a string
        if self.iterable_test_file(test_file):
            for t in test_file:
                self.predict(t, range = range)
            return

        # param use_dataset_ratios_for_accumulated_probabilities: only important when TRAIN_PER_FILE == True
        use_dataset_ratios_for_accumulated_probabilities = self.pipeline_configuration["USE_DATASET_RATIOS_FOR_ACCUMULATED_PROBABILITIES"]

        super().predict(test_file, range = range)

        if not self.pretrained_mode:
            self.selected_feature_cols, self.feature_transformer = self.get_selected_features_and_transformer(self.train_files)

        window_size = np.max(self.pipeline_configuration["WINDOW_SIZES"])
        # test_dataset_range = self.pipeline_configuration["TEST_RANGE"]
        step_size_test = self.pipeline_configuration["STEP_SIZE_TEST"]
        samling_mode = self.pipeline_configuration["SAMPLING_MODE"]
        if self.pipeline_configuration["STEP_SIZE_TEST"] == 1:
            sampling_mode = "fixed"

        loc_indices_list, loc_indices_files_list = get_loc_indices_list_from_files([test_file], step_size = step_size_test, window_size=window_size, sampling_mode=samling_mode, dataset_range = range, feature_file = True)
        loc_indices = loc_indices_list[0]
        
        X_test, Y_test = get_data_labels_from_file(test_file, selected_feature_cols = self.selected_feature_cols, is_training = False, data_transformer = self.feature_transformer, loc_indices=loc_indices)

        self.save_dataset_distribution_from_labels(Y_test, self.prediction_output_path, dataset_type = "test")

        if os.path.exists(self.prediction_output_path):
            self.predictions = super().get_predictions_from_file(self.prediction_output_path)
            print(f"Loaded predictions from file: {test_file}.")
            self.probabilities = self.get_probabilities_from_file(self.probabilities_output_path)
            return self.predictions, self.probabilities

        if self.pipeline_configuration["TRAIN_PER_FILE"]:
            # training per file
            test_basename = os.path.basename(test_file).split(".")[0]
            probabilities_output_dir_per_testfile = os.path.join(self.predictions_output_dir, f"probabilities_individual_on_{test_basename}")
            if range != (0, 1.0): 
                probabilities_output_dir_per_testfile += f"_({range[0]},{range[1]})"

            if not os.path.exists(probabilities_output_dir_per_testfile):
                os.makedirs(probabilities_output_dir_per_testfile)

            for classifier_filepath in self.classifier_paths:
                
                model_probabilities_out_fname = os.path.join(probabilities_output_dir_per_testfile, os.path.basename(classifier_filepath).replace(".pkl", "_probabilities.pkl"))
                
                if os.path.exists(model_probabilities_out_fname):
                    continue
                
                model = load_from_pickle(classifier_filepath)
                Y_pred_prob = predict_probability(model, X_test)
                Y_pred_prob.index = loc_indices
                if INDEX_COL not in Y_pred_prob.columns.values:
                    Y_pred_prob[INDEX_COL] = Y_pred_prob.index
                save_as_pickle(Y_pred_prob, model_probabilities_out_fname) #, index = False)

            if use_dataset_ratios_for_accumulated_probabilities:
                self.classifier_dataset_dict = load_from_pickle(self.classifier_dataset_size_f)
                combine_probability_predictions(probabilities_output_dir_per_testfile, self.probabilities_output_path, self.classifier_dataset_size_dict)
            else: 
                combine_probability_predictions(probabilities_output_dir_per_testfile, self.probabilities_output_path)
            self.predictions = probabilities_from_path_to_classification(self.probabilities_output_path) #, self.prediction_output_path)
            
        else:
            # If train per file = False --> model is saved in self.model
            # load model and make prediction
            Y_pred_prob = predict_probability(self.model, X_test)
            Y_pred_prob.index = loc_indices
            if INDEX_COL not in Y_pred_prob.columns.values:
                Y_pred_prob[INDEX_COL] = Y_pred_prob.index
            save_as_pickle(Y_pred_prob, self.probabilities_output_path)
            
            self.predictions = probabilities_from_path_to_classification(self.probabilities_output_path) #, self.prediction_output_path)

        save_as_pickle(self.predictions, self.prediction_output_path) 

        print(f"Done predictions for {test_file}.")

        self.probabilities = self.get_probabilities_from_file(self.probabilities_output_path)

        return self.predictions, self.probabilities

    def evaluate(self, test_file, gt_file = None, label_mapping = None, range = (0, 1.0)):
        '''
        :param test_file: the file containing the features at specific indices 
        :param gt_file: the ground truth file
        '''
        super().evaluate(test_file)

        gt_values, prediction_values, probabilities_df =  super().get_ground_truth_prediction_values(test_file, gt_file, range = range, label_mapping = label_mapping)

        accuracy, f1_score, cross_entropy_loss = super().evaluate_from_prediction_gt_prob_values(gt_values, prediction_values, probabilities_df)

        super().evaluation_to_statistic_file()

        return accuracy, f1_score, cross_entropy_loss 

    def visualize_prediction_vs_gt(self, test_file, gt_file):
        '''
        so far this only is possible for the case STEP_SIZE_TEST == 1
        For the shallow approach the ground truth data is not in the test file 
        but in a separate ground truth file. The shallow test file contains the 
        feature vectors (and not the original signals).
        '''
        # assert(self.pipeline_configuration["STEP_SIZE_TEST"] == 1)
        assert(os.path.exists(gt_file))
        self.predict(test_file)
        super().visualize_prediction_vs_gt(self.prediction_output_path,
                gt_file,
                signals = ["JOINT_ANGLE", "DDD_ACC_TOTAL", LABEL_COL, PREDICTION_COL],
                data_scaling_factors = [1, 1, 1/10, 1/10])


def get_classifier(classifier_type, classifier_config, pipeline_config, basename = "", pretrained_path = "", output_dir = "", selected_features_path = "", transformer_path = ""):
    '''
    Based on the classifier type, classifier config and pipeline_config return 
    either a NeuralNetwork or ShallowClassifier-Instance - both of which are 
    children of the Classifier class.
    :param classifier_config: either a dictionary with the configuration or a path to the pickle file
    :param pipeline_config: either a dictionary with the configuration or a path to the pickle file
    '''
    if type(classifier_config) == str:
        assert(classifier_config.endswith(".pkl"))
        if os.path.exists(classifier_config):
            classifier_config = load_from_pickle(classifier_config)

    if type(pipeline_config) == str:
        assert(pipeline_config.endswith(".pkl"))
        if os.path.exists(pipeline_config):
            pipeline_config = load_from_pickle(pipeline_config)

    if len(pretrained_path) > 0:
        if os.path.exists(pretrained_path):
            pipeline_config["PRETRAINED"] = pretrained_path

            if classifier_type in SHALLOW_IMPLEMENTATIONS:
                assert(os.path.exists(selected_features_path))
                pipeline_config["FEATURE_COLS"] = selected_features_path
                
                if pipeline_config["FEATURE_SELECTION_METHOD"] in ["pca"]:
                    assert(os.path.exists(transformer_path))
                    pipeline_config["TRANSFORMER"] = transformer_path

    if classifier_type.upper() in NN_IMPLEMENTATIONS:
        classifier = NeuralNetwork(classifier_type, classifier_config, pipeline_config, basename, output_dir)
    elif classifier_type.lower() in SHALLOW_IMPLEMENTATIONS:
        classifier = ShallowClassifier(classifier_type, classifier_config, pipeline_config, basename, output_dir)
    else:
        print(f"Invalid classifier type: {classifier_type}")
        sys.exit()
    
    return classifier

def get_default_classifier_config(classifier_type):
    '''
    Based on the classifier type return the default configuration.
    '''
    if classifier_type in NN_IMPLEMENTATIONS:
        config = get_network_config(classifier_type)
        return config
    elif classifier_type.lower() in SHALLOW_IMPLEMENTATIONS:
        config = get_shallow_classifier_config(classifier_type)
        return config
    else:
        print(f"Invalid classifier type: {classifier_type}")
        sys.exit()

def get_default_pipeline_config(classifier_type):
    '''
    
    '''
    if classifier_type in NN_IMPLEMENTATIONS:
        config = NN_CONFIG
        # return NN_CONFIG
    elif classifier_type.lower() in SHALLOW_IMPLEMENTATIONS:
        config = SHALLOW_CONFIG
    else:
        print(f"Invalid classifier type: {classifier_type}")
        print(f"Valid NN classifiers: {NN_IMPLEMENTATIONS}")
        print(f"Valid shallow (feature based) classifier: {SHALLOW_IMPLEMENTATIONS}")
        sys.exit()

    return config
