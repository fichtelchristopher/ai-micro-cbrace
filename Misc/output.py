'''

Saving results etc
'''
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

def print_to_eval_overview_file(eval_overview_output_f, f1_scores, accuracies, results_dir):
    '''
    Prints an overview of mean f1 scores to a csv file
    
    :param eval_overview_output_f: path to a csv file, where overviewwill be saved
    :param f1_scores: list, the f1 scores associated to a test file
    :parm test_file_names: list of strings, corresponding test file for a f1 score 

    '''
    if os.path.exists(eval_overview_output_f):
        eval_df = pd.read_csv(eval_overview_output_f)
    else:
        eval_df = pd.DataFrame(columns=["results_dir", "n_test_files", "f1_mean", "accuracy_mean"])
    
    if results_dir not in eval_df["results_dir"]:
        eval_df = eval_df.append({"results_dir": results_dir, "n_test_files": len(f1_scores), "f1_mean": np.mean(f1_scores), "accuracy_mean": np.mean(accuracies)}, ignore_index = True)
        eval_df.to_csv(eval_overview_output_f, index=False)



def print_optimization_trial_results_to_file(output_f, input_files, test_files, f1_scores, accuracies):
    '''
    Prints optimization trial results (f1 scores) to text file
    :param output_f: a text file for output
    :param input_files: the training files
    :param test_files: list of test files, test file means that training has been done on all input files except the test file and evaluation done on test file (LOSO manner)
    :param f1_scores: the corresponding f1 test scores, list
                        list has to have same length as the test_files list
    '''
    assert(len(test_files) == len(f1_scores) == len(accuracies))

    original_stdout = sys.stdout
    with open(output_f, "w") as f:
        sys.stdout = f # Change the standard output to the file we created.
        print("\nInput files")
        for input_file in input_files:
            print(input_file)
        print("\nTest files")

        for test_file, f1_score, acc in zip(test_files, f1_scores, accuracies):
            print(f"{test_file}, \tf1_score: {f1_score}\taccuracy: {acc}")
        
        f1_scores_mean = np.mean(f1_scores)
        print(f"\nMean f1 score: {f1_scores_mean}")
        accuracy_mean = np.mean(accuracies)
        print(f"\nMean f1 score: {accuracy_mean}")
        
        sys.stdout = original_stdout

def print_configurations_to_file(output_file, configurations: list):
    '''
    This writes a list of configuarations to the output_file. 
    A configuration has to be a dictionary. 

    '''
    if os.path.exists(output_file):
       return

    if not os.path.exists(Path(output_file).parent):
        os.makedirs(Path(output_file).parent)
    
    original_stdout = sys.stdout
    
    with open(output_file, "w") as f:
        sys.stdout = f # Change the standard output to the file we created.
        for dict_ in configurations:
            keys = list(dict_.keys())
            keys.sort()
            
            for k in keys: 
                print(f"{k}:\t\t{dict_[k]}")
            print("\n\n")
    sys.stdout = original_stdout