''' 
Master thesis project of Christopher Fichtel (christopher.fichtel@ottobock.com) started in Oct 2022
AI in operating ortheses

File created 11 Oct 2022

Python Code Base for an activity  
'''
from Configs.namespaces import *
import numpy as np
from Misc.utils import *
from Processing.loading_saving import load_from_pickle, save_as_pickle

activity_int_name_dict = get_reversed_dictionary()

class AOS_Activity:
    '''
    An AOS_activity contains the activity type 
    '''

    def __init__(self, aos_activity_df, activity_progression):
        '''
        :param aos_activity_df:     the dataframe for this activity already indexed i.e. all data corresponds to "same activity" (this condition will be checked via assert statement) 
        :return: 
        '''
        assert(len(set(list(activity_progression))) == 1)
        assert(len(aos_activity_df) == len(list(activity_progression)))

        self._code = int(activity_progression[0])                # integer representing the activity

        self.activity_df = aos_activity_df[AOS_ACTIVITY_COLS]

        self.FH_min = None
        self.FH_max = None
        self.Mk_min = None
        self.Mk_max = None
        self.K_max  = None

        self.generate_activity()

    def generate_activity(self):
        '''
        Save important columns of the dataframe (AOS_activity_cols) from the namespaces file.
        '''
        self.FH_min = np.min(self.activity_df["JOINT_LOAD"].values)
        self.FH_max = np.max(self.activity_df["JOINT_LOAD"].values)

        self.K_max = np.max(self.activity_df["JOINT_ANGLE"])
        
        # M_K=AOSdata(:,col.FH)*cal_FH.*knee_lever(int16(AOSdata(:,col.K)*10)+96,2) --> knee moment
        self.Mk_min = np.min(self.activity_df[KNEE_MOMENT_COL].values)
        self.Mk_max = np.max(self.activity_df[KNEE_MOMENT_COL].values)

    def _get_possible_signal_values(self):
        '''
        
        '''
        return self.activity_df.columns.values

    def _get_signal(self, signal: str):
        '''
        Return the time series for the given signal. 

        :param signal: str, has to be in the "AOS_ACTIVITY_COLS" constant defined in namespaces.py (12 Oct 2022)

        '''
        assert (signal in self.activity_df.columns.values)

        return self.activity_df[signal].values
        

    def plot_signal(self, signal: str, clear_figure = True):
        '''
        Plot the given signal 
        '''
        assert (signal in self.activity_df.columns.values)

        plt.plot(self._get_signal(signal))
        plt.show()
    
    def plot_signals(self, signal_list, clear_figure = True):
        '''
        Plot the list of signals
        '''
        assert (np.all(signal in self._get_possible_signal_values() for signal in signal_list))

        for signal in signal_list:
            plt.plot(self._get_signal(signal))

        plt.show()