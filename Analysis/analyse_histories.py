'''
Given an input directory, recursively iteratate over the histories and write out the values in one csv file. 
'''
import glob
from Processing.loading_saving import load_from_pickle
import pandas as pd
import os
import numpy as np



def main(input_dir, output_f):


    history_files = glob.glob(f"{input_dir}/**/*history.pkl", recursive=True)

    eval_metrics = ["accuracy", "f1", "loss", "val_accuracy", "val_f1", "val_loss"] 

    if os.path.exists(output_f):
        history_df = pd.read_csv(output_f)
    else:
        history_df = pd.DataFrame(columns = ["history_path"] + eval_metrics)

    for h in history_files:

        history_dict = load_from_pickle(h)


        if not set(eval_metrics).issubset(set(list(history_dict.keys()))):
            continue

        if type(history_dict["val_loss"]) == list:
            # get the best --> this is the model that has been saved according to the fitting method
            best_idx = np.array(history_dict["val_loss"]).argmin()

            for k in history_dict.keys():
                history_dict[k] = history_dict[k][best_idx]

        # Delete entries that are not important    
        delete_entry_keys = [k for k in history_dict.keys() if k not in eval_metrics]     
        for k in delete_entry_keys:
            del history_dict[k]

        history_dict["history_path"] = h

        if h not in history_df["history_path"]:

            history_df = history_df.append(history_dict, ignore_index = True)

    history_df.to_csv(output_f, index = False)

    print(f"Wrote {output_f}")



if __name__ == "__main__":
    input_dir = "C:/Users/FICHTEL/Ottobock SE & Co. KGaA/Deep Learning - General/AI-Pipeline/aos/DS2/05-Classifier"
    output_f = f"{input_dir}/history_evaluation.csv"
    main(input_dir, output_f)

    for classifier_type in ["CNN_SIMPLE", "LSTM_SIMPLE", "CNN_LSTM_SIMPLE", "xgboost", "svm", "lda", "rf"]:
        classifier_input_dir = input_dir + f"/{classifier_type}"  
        classifier_f = f"{classifier_input_dir}/history_evaluation.csv"
        main(classifier_input_dir, classifier_f)

    
    
    