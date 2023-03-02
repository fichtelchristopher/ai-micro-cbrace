'''
Little script to remove empty folders in a given directory.
'''

import os

def remove_empty_folders(path_abs):
    walk = list(os.walk(path_abs))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            os.rmdir(path)

if __name__ == '__main__':
    path = "C:/Users/fichtel/Ottobock SE & Co. KGaA/Deep Learning - General/AI-Pipeline/aos/06-Parameter_Optimization/CNN_LSTM_SIMPLE"
    remove_empty_folders(path)