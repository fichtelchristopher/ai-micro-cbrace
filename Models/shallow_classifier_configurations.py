"""
Defines the standard parameters for the models under neural_networks.py
"""

def get_shallow_classifier_config(model_type):
    '''
    :param model_type: can be "cnn" "lstm" or "cnn_lstm"

    :return: the configuration
    '''
    model_type = model_type.lower()
    if model_type == "svm":
        return SVM_CONFIG
    if model_type == "sgd":
        return SGD_CONFIG
    elif model_type == "xgboost":
        return XGBOOST_CONFIG
    elif model_type == "rf":
        return RF_CONFIG
    elif model_type == "lda":
        return LDA_CONFIG
    else:
        print(f"Invalid model type {model_type}.")
        return None


LDA_CONFIG = {
    "classifier": "lda",
    "solver": "svd", 
    "priors": None,
    "n_components": None,
    "tol": 1.0e-4,
    "shrinkage": None
}

XGBOOST_CONFIG = {
    "classifier": "xgboost",
    "booster": "gbtree",
    "validate_parameters": False,
    "eta" : 0.3, 
    "gamma" : 0,
    "max_depth" : 6,
    "min_child_weight" : 1, 
    "max_delta_step" : 0,
    "subsample" : 1,
    "lambda": 1,
    "alpha" : 0
}

SVM_CONFIG = {
    "classifier": "svm",
    "kernel": "rbf",
    "gamma": "scale",                    
    "shrinking": True,
    "tol": 1e-3,
    "class_weight": None,
    "decision_function_shape": "ovr",
    "probability": True
}

SGD_CONFIG = {
    "classifier": "sgd",
    "loss": "hinge",
    "penalty": "l2",
    "alpha": "0.0001",
    "max_iter" : 1000,
    "tol" : 1e-3
}

RF_CONFIG = {
    "classifier": "rf",
    "n_estimators":  100,
    "criterion": "gini", 
    "max_depth": None, 
    "min_samples_split": 2,
    "min_samples_leaf":  1, 
    "min_weight_fraction_leaf": 0.0, 
    "max_features": "sqrt",
    "oob_score": False,
    "class_weight": None, 
}