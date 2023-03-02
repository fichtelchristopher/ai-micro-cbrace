from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.cluster import KMeans

from Models.shallow_classifier_configurations import *

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def get_svm(svm_config = None):
    '''
    SVM 

    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html 

    Batchwise SVM https://stackoverflow.com/questions/40737750/scikit-learn-svm-with-a-lot-of-samples-mini-batch-possible 
    https://stackoverflow.com/questions/23056460/does-the-svm-in-sklearn-support-incremental-online-learning/43801000#43801000 
    '''
    if type(svm_config) == type(None):
        print("Loading default sgd configuration.")
        svm_config = SVM_CONFIG

    clf = SVC(
        kernel                  = svm_config["kernel"],
        gamma                   = svm_config["gamma"],
        shrinking               = svm_config["shrinking"],
        tol                     = svm_config["tol"],
        class_weight            = svm_config["class_weight"],
        decision_function_shape = svm_config["decision_function_shape"],
        probability = True)

    return clf

def get_lda(lda_config = None):
    '''
    https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html 
    '''
    if type(lda_config) == type(None):
        print("Loading default lda configuration.")
        lda_config = LDA_CONFIG

    if lda_config["solver"] == "svd":
        shrinkage = None
    else:
        shrinkage = "auto"

    clf = LDA(
        solver          = lda_config["solver"], 
        priors          = lda_config["priors"],
        n_components    = lda_config["n_components"],
        tol             = lda_config["tol"],
        shrinkage       = shrinkage)

    return clf

def get_rf(rf_config = None):
    '''
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
    '''
    if type(rf_config) == type(None):
        print("Loading default rf configuration.")
        rf_config = RF_CONFIG

    clf = RandomForestClassifier(
        n_estimators= rf_config["n_estimators"],
        criterion= rf_config["criterion"],
        max_depth =  rf_config["max_depth"],
        min_samples_split= rf_config["min_samples_split"],
        min_samples_leaf = rf_config["min_samples_leaf"],
        min_weight_fraction_leaf= rf_config["min_weight_fraction_leaf"],
        max_features = rf_config["max_features"],
        oob_score= rf_config["oob_score"],
        class_weight= rf_config["class_weight"]
    )

    return clf


'''
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier

Linear classifiers (SVM, logistic regression, etc.) with SGD training.

This estimator implements regularized linear models with stochastic gradient descent (SGD) learning: the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (aka learning rate).
SGD allows minibatch (online/out-of-core) learning via the partial_fit method.
For best results using the default learning rate schedule, the data should have zero mean and unit variance.

This implementation works with data represented as dense or sparse arrays of floating point values for the features.
The model it fits can be controlled with the loss parameter; by default, it fits a linear support vector machine (SVM).

'''
def get_sgd(sgd_config = None):
    '''
    
    '''
    if type(sgd_config) == type(None):
        print("Loading default sgd configuration.")
        sgd_config = SGD_CONFIG

    clf = SGDClassifier(
        loss=sgd_config["loss"],
        penalty=sgd_config["penalty"],
        alpha = sgd_config["alpha"],
        max_iter=sgd_config["max_iter"],
        tol = sgd_config["tol"])

    return clf


def get_xgboost(xgboost_config = None):
    '''
    Xgboost: an ensemble decision tree
    Advantage: faster training and prediciton times
    https://journals.sagepub.com/doi/full/10.1177/09544119211001238?casa_token=xc-jnG0jjA4AAAAA%3AENqhYTwfYtl0dHSCORH61_57E3ADKiVxl9p3WIIG3aqiNkmYm9PdR7ieOEfeCOmYcmJRmAV8ssO3 
    
    https://en.wikipedia.org/wiki/XGBoost
    '''
    if type(xgboost_config) == type(None):
        print("Loading default xgboost configuration.")
        xgboost_config = XGBOOST_CONFIG

    clf = XGBClassifier(
        booster = xgboost_config["booster"],
        validate_parameters = xgboost_config["validate_parameters"],
        eta = xgboost_config["eta"],
        gamma = xgboost_config["gamma"],
        max_depth = xgboost_config["max_depth"],
        min_child_weight = xgboost_config["min_child_weight"],
        max_delta_step = xgboost_config["max_delta_step"],
        subsample = xgboost_config["subsample"],
        # sampling_method = xgboost_config["sampling_method"],
        kwargs={"lambda": xgboost_config["lambda"]},
        alpha = xgboost_config["alpha"]
         )

    return clf


def get_kmeans(n_classes = 3, kmeans_config = None):
    '''
    Kmeans clustering
    '''
    clf = KMeans(n_clusters = n_classes)

    return clf