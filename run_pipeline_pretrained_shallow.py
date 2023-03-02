from classifier import *
import pickle
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from Processing.loading_saving import *

classifier_dir = "C:/Users/FICHTEL/Ottobock SE & Co. KGaA/Deep Learning - General/AI-Pipeline/leipzig/05-Classifier/xgboost/s_10_w_10_30_50_75_100_numf75_C19"

# subject = "HK0203" #0.95
subject = "DM0104"
# subject = "WF0101"

n_features = 75

classifier_f = os.path.join(classifier_dir, f"Training_10/{subject}_normalized_leave_out.pkl")
transformer_f = os.path.join(classifier_dir, f"Training_10/{subject}_normalized_leave_out_transformer_{n_features}.pkl")
selected_features_cols_f = os.path.join(classifier_dir, f"Training_10/{subject}_normalized_leave_out_selected_features_{n_features}.pkl")

pipeline_config_f = os.path.join(classifier_dir, "pipeline_config.pkl")
classifier_config_f = os.path.join(classifier_dir, "classifier_config.pkl")

with open(pipeline_config_f, "rb") as f:
    pipeline_config = pickle.load(f)

with open(classifier_config_f, "rb") as f:
    classifier_config = pickle.load(f)

pipeline_config["PRETRAINED"]       = classifier_f
pipeline_config["TRANSFORMER"]      = transformer_f
pipeline_config["FEATURE_COLS"]     = selected_features_cols_f
pipeline_config["STEP_SIZE_TEST"]   = 1

leipzig_test_dir = "C:/Users/fichtel/Ottobock SE & Co. KGaA/Deep Learning - General/AI-Pipeline/leipzig/04-Features-Normalized"
test_file =  leipzig_test_dir +  f"/{subject}_normalized.csv"
leipzig_gt_dir = "C:/Users/fichtel/Ottobock SE & Co. KGaA/Deep Learning - General/AI-Pipeline/leipzig/01-Activities-GT-Full"
leipzig_gt_dir = "C:/Users/fichtel/Ottobock SE & Co. KGaA/Deep Learning - General/AI-Pipeline/leipzig/02-Activities-GT"
gt_file_basename = os.path.basename(test_file).replace("_normalized.csv", ".csv")
gt_file = leipzig_gt_dir + f"/{gt_file_basename}" 

classifier = ShallowClassifier("xgboost",
                pipeline_configuration = pipeline_config,
                classifier_configuration = classifier_config,
                classifier_output_dir = classifier_dir,
                classifier_name = "xgboost_pretrained")

classifier.predict(test_file)
f1_score, accuracy = classifier.evaluate(test_file)
print(f"{os.path.basename(test_file)}: F1={f1_score}, accuracy={accuracy}")
classifier.visualize_prediction_vs_gt(gt_file)
# plt.title(subject)
plt.show()
print("")