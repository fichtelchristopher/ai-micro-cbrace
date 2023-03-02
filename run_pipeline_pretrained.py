from classifier import *
import pickle
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from Analysis.transition_probability_analysis import *
from Processing.loading_saving import *
from Processing.file_handling import get_leipig_files_from_dir, get_aos_files_from_dir


def get_pretrained_classifier(classifier_dir, classifier_name, classifier_type):
    '''
    :param nn_pretrained_path: only when the classifier type is in the NN_IMPLEMENTATIONS list. 
    '''
    # Load the configurations & set parameters
    if classifier_type in SHALLOW_IMPLEMENTATIONS:
        classifier_path = glob.glob(classifier_dir + f"/**/{classifier_name}.pkl", recursive=False)
        assert(os.path.exists(classifier_path))
        classifier = load_from_pickle(classifier_path)
    else:
        nn_pretrained_paths = glob.glob(classifier_dir + f"/**/{classifier_name}*.h5", recursive=True)
        assert(len(nn_pretrained_paths) == 1)

        #load nn network from configs
        classifier_config = load_from_pickle(classifier_dir + "/classifier_config.pkl")
        pipeline_config = load_from_pickle(classifier_dir + "/pipeline_config.pkl")
    
        classifier = get_classifier(classifier_type, classifier_config, pipeline_config, basename = "", pretrained_path=nn_pretrained_paths[0]) #, output_dir = classifier_output_dir)

        activity_code_type_dict_f = glob.glob(classifier_dir + f"/**/activity_code_type_dict.pkl", recursive=True)
        classifier.activity_code_type_dict = load_from_pickle(activity_code_type_dict_f[0])                 

    classifier.set_classifier_output_dir()

    return classifier

def main(classifier: Classifier, classifier_type, test_database, test_dataset, transition_analysis = True, label_mapping = None):


    # activities_dir = f"C:/Users/fichtel/Desktop/{db}/02-Activities-GT"
    full_activities_dir =   AI_PIPELINE_DIR + f"/{test_database}/01-Activities-GT-Full"
    test_activities_dir =   AI_PIPELINE_DIR + f"/{test_database}/DS{test_dataset}/02-Activities-GT"
    test_features_dir   =   AI_PIPELINE_DIR + f"/{test_database}/DS{test_dataset}/04-Features-Normalized"
    test_files = get_train_test_files(classifier_type, activities_output_dir = test_activities_dir, features_normalized_dir = test_features_dir, database = test_database)

    if len(test_files) == 0:
        full_activities_dir =   full_activities_dir.replace(AI_PIPELINE_DIR, "C:/Users/fichtel/Ottobock SE & Co. KGaA/Deep Learning - General/AI-Pipeline")     # for visualising the signals 
        test_activities_dir =   test_activities_dir.replace(AI_PIPELINE_DIR, "C:/Users/fichtel/Ottobock SE & Co. KGaA/Deep Learning - General/AI-Pipeline")     # for getting the transitions 
        test_features_dir   =   test_features_dir.replace(AI_PIPELINE_DIR, "C:/Users/fichtel/Ottobock SE & Co. KGaA/Deep Learning - General/AI-Pipeline")       # for getting the transitions
        test_files = get_train_test_files(classifier_type, activities_output_dir = test_activities_dir, features_normalized_dir = test_features_dir, database = test_database)

    # Get the overall test accuracy
    overall_test_acc, overall_test_f1_score, overall_test_ce_loss = classifier.evaluate(test_files, label_mapping = None)

    if not transition_analysis:
        return overall_test_acc, overall_test_f1_score, overall_test_ce_loss 

    f1_scores = []
    ce_losses = []
    for test_file in test_files:
        classifier.classifier_name = os.path.basename(test_file).replace(".pkl", "")
        gt_df = load_from_pickle(test_file)
        predictions, probabilities = classifier.predict(test_file)

        test_acc, test_f1_score, test_ce_loss = classifier.evaluate(test_file, label_mapping = label_mapping)
        print(f"{os.path.basename(test_file)}: accuracy={test_acc}, F1={test_f1_score}, loss = {test_ce_loss}")
        f1_scores.append(test_f1_score)
        ce_losses.append(test_ce_loss)

        visualize_predictions = False
        if visualize_predictions:
            classifier.visualize_prediction_vs_gt(test_file, label_mapping = label_mapping)
            print(classifier.f1_score)

        # Get the corresponding data file
        corresponding_signal_file = full_activities_dir + f"/{os.path.basename(test_file)}" # in case of shallow implementations, the test file does not hold the raw signal data  
        corresponding_signal_file = corresponding_signal_file.replace("_gt_features_normalized.pkl", ".pkl")
        corresponding_signal_file = corresponding_signal_file.replace("_gt_reduced.pkl", ".pkl")
        data_df = load_from_pickle(corresponding_signal_file)

        # from code and to code have to be chosen based on the activity code type dict 
        transition_analysis(probabilities, predictions, gt_df, data_df, classifier.activity_code_type_dict , from_code = -1, to_code = 200, transition_range = 200, output_dir = "")  

    print(f"f1-scores: {f1_scores}")
    print(f"mean = {np.mean(f1_scores)}\n")

    print(f"ce-losses: {ce_losses}")
    print(f"loss-mean = {np.mean(ce_losses)}\n")

    return overall_test_acc, overall_test_f1_score, overall_test_ce_loss 


if __name__ == "__main__":

    # # Either set manually here or in the loop as below
    # classifier_type = "CNN_LSTM_SIMPLE"
    # step = 250
    # classifier_dir = f"{AI_PIPELINE_DIR}/aos/DS1/05-Classifier/CNN_LSTM_SIMPLE/s{step}_w32_e100_lr0.001_b64_C1"
    # classifier_trained_dir = classifier_dir + f"/Training_{step}/CNN_LSTM_SIMPLE_6_train_files"
    # classifier_pretrained_path = classifier_trained_dir + f"/CNN_LSTM_SIMPLE_6_train_files.h5"
    # classifier_output_dir = classifier_dir + "/LeipzigPredictions"
    # overall_test_acc, overall_test_f1_score, overall_test_ce_loss = main(classifier_type, classifier_dir, classifier_pretrained_path, classifier_output_dir, test_database, test_dataset, transition_analysis = False)

    # Set the test database
    test_database   = "leipzig"
    test_dataset    = 1
    classifier_base_dir = f"{AI_PIPELINE_DIR}/aos/DS1/05-Classifier" # this will also be used for saving the statistics file
    label_mapping = None

    stat_output_f = classifier_base_dir + f"/predictions_on_{test_database}_dataset{test_dataset}.csv"
    if os.path.exists(stat_output_f):
        stat_df = pd.read_csv(stat_output_f)
    else:
        stat_df = pd.DataFrame(columns = ["classifier_path", "test_db", "test_dataset", "test_acc", "test_f1", "test_ce_loss"])


    # Run the loop 
    for classifier_type in NN_IMPLEMENTATIONS: # + SHALLOW_IMPLEMENTATIONS: #"CNN_LSTM_SIMPLE"]: #NN_IMPLEMENTATIONS: # + SHALLOW_IMPLEMENTATIONS:
        default_classifier_dir = f"{classifier_base_dir}/{classifier_type}"
        for step in [250]: #, 500, 1000, 2500]:
            for subdir in os.listdir(default_classifier_dir):
                if not ((f"s_{step}_" in subdir) or (f"s{step}_" in subdir)):
                    continue
                
                # get the classifier
                classifier_dir = f"{default_classifier_dir}/{subdir}"
                classifier_name = f"{classifier_type}_6_train_files"
                output_dir = classifier_dir + "/LeipzigPredictions"
                try:
                    classifier = get_pretrained_classifier(classifier_dir, classifier_name, classifier_type, output_dir)
                except:
                    continue
                
                classifier.pipeline_config["STEP_SIZE_TEST"] = 1
                if classifier_type in NN_IMPLEMENTATIONS:
                    classifier.pipeline_config["TRAIN_PER_FILE"] = False
                classifier.classifier_name = "AOS_CNN_LSTM_on_Leipzig"

                test_acc, test_f1, test_ce_loss = main(classifier, classifier_type, test_database, test_dataset, transition_analysis = False, label_mapping = True)  

                results_dict = {
                    "classifier_path": classifier.classifier_name,
                    "test_db": test_database,
                    "test_dataset": test_dataset,
                    "test_acc": test_acc,
                    "test_f1": test_f1,
                    "test_ce_loss": test_ce_loss
                }

                stat_df = stat_df.append(results_dict, ignore_index = True)

                stat_df.to_csv(stat_output_f, index = False)

                print(f"Wrote {stat_df}")

