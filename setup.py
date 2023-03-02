'''
In order for your python installation to find all required paths run:
pip install -e . when being in the base "ai-prothesis-operation" directory
'''

from setuptools import setup, find_packages, find_namespace_packages

setup(name='ai-prothesis-operation', version='1.0', packages=find_packages())               # all directories with an init file
setup(name='ai-prothesis-operation', version='1.0', packages=find_namespace_packages())     # all directories independent of if they have an init file or not
