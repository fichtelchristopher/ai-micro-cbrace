'''
Import this file so that matplotlib uses custom default font.

You will have to change the path names.
'''
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Usetex enables to use mathematical formulations
# Setup following tutorial on https://omz-software.com/pythonista/matplotlib/users/usetex.html
# https://miktex.org/download
# plt.rcParams['text.usetex'] = True

# The default font we want to use.
font_path = "//ATWISRV1/GRUPPEN/AR/Studenten/Studenten_2022/Christopher_Fichtel/thesis/font/lmroman7-regular.otf"
if not os.path.exists(font_path):
    font_path = "C:/Users/fichtel/Desktop/font/lmroman7-regular.otf"

font_prop = fm.FontProperties(fname=font_path)

fm.fontManager.addfont(font_path)
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name() 


