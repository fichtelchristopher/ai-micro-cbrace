"""
Defines the standard parameters for the models under neural_networks.py
"""

def get_network_config(model_type):
    '''
    :param model_type: can be "cnn" "lstm" or "cnn_lstm"

    :return: the configuration
    '''
    model_type = model_type.lower()
    if model_type == "cnn":
        return CNN_NETWORK_CONFIG
    if model_type == "lstm":
        return LSTM_NETWORK_CONFIG
    elif model_type == "cnn_lstm":
        return CNN_LSTM_NETWORK_CONFIG
    elif model_type == "lstm_simple":
        return LSTM_SIMPLE_NETWORK_CONFIG
    elif model_type == "cnn_simple":
        return CNN_SIMPLE_NETWORK_CONFIG
    elif model_type == "cnn_lstm_simple":
        return CNN_LSTM_SIMPLE_NETOWRK_CONFIG
    else:
        print(f"Invalid model type {model_type}.")
        return None

CNN_NETWORK_CONFIG = {
                "architecture": "CNN",
                "layers_number": 1,
                "n_filters_conv_0": 256, 
                "pool_size": 2,
                "conv_filter_size": 4,
                "n_filters_dense": 4,

                "dropout": 0.5,

                "batch_normalization": True
                }

CNN_LSTM_NETWORK_CONFIG = {
            "architecture": "CNN_LSTM",
            "conv_layers_number": 2,
            "n_filters_conv_0": 128, 
            "n_filters_conv_1": 64, 
            "conv_filter_size": 3,
            "pool_size": 2,

            "lstm_layers_number": 1,
            "n_filters_lstm_0": 8,

            "dropout": 0.5,

            "dense_layers_number": 2,
            "n_filters_dense_0": 16,
            "n_filters_dense_1": 8,

            "batch_normalization": True
            }


LSTM_NETWORK_CONFIG = {
            "architecture": "LSTM",
            "lstm_layers_number": 1,
            "n_filters_lstm_0": 8,

            "dropout": 0.5,

            "dense_layers_number": 2,
            "n_filters_dense_0": 16,
            "n_filters_dense_1": 8,

            "batch_normalization": True

            }

# Only to have a config, no real info yet
LSTM_SIMPLE_NETWORK_CONFIG = {
    "architecture": "LSTM_SIMPLE",
    "lstm_units": 128, 
    "dropout": 0.25,
    "activation": 'softmax'
}

CNN_SIMPLE_NETWORK_CONFIG = {
    "architecture": "CNN_SIMPLE",
    "n_layers_conv": 2,             # can be either 1 or 2 (25.1.2023)
    "n_filters_conv": 64,
    "kernel_size_conv": 3,  

    "pool_size": 2,
    "pool_type": "max",        

    "dense_units": 64,
    "dropout": 0.5,
    "activation": 'softmax'
}

CNN_LSTM_SIMPLE_NETOWRK_CONFIG = {
    "architecture": "CNN_LSTM",
    "n_layers_conv": 2,             # can be either 1 or 2 (25.1.2023)
    "n_filters_conv": 64,
    "kernel_size_conv": 3,

    "pool_size": 2,
    "pool_type": "max",

    "lstm_units": 128, 
    "conv_dropout": 0.3,
    "lstm_dropout": 0.1,
    "activation": 'softmax'
}
