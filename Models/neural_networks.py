
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, AveragePooling1D
from tensorflow.keras import regularizers
from keras.layers import Input, LSTM, Dense, Dropout

from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import Dense, Dropout
from keras.models import Sequential


from Models.neural_networks_configurations import CNN_LSTM_NETWORK_CONFIG, CNN_NETWORK_CONFIG, LSTM_NETWORK_CONFIG, LSTM_SIMPLE_NETWORK_CONFIG, CNN_LSTM_SIMPLE_NETOWRK_CONFIG, CNN_SIMPLE_NETWORK_CONFIG

from keras.models import Model

def get_cnn(input_shape, n_classes_out, cnn_network_config = None):  

    '''
    Standard CNN
    '''
    if type(cnn_network_config) == type(None):
        print("Loading default cnn configuration.")
        cnn_network_config = CNN_NETWORK_CONFIG

    n_timesteps = input_shape[0]
    n_features = input_shape[1]
    model = Sequential()

    model.add(tf.keras.layers.Input(shape = (n_timesteps, n_features)))
    
    for i in range(cnn_network_config['layers_number']):

        model.add(Conv1D(filters=int(cnn_network_config[f"n_filters_conv_{i}"]), kernel_size= int(cnn_network_config['conv_filter_size']), activation='relu', padding = "same"))  #425
        model.add(MaxPooling1D(pool_size=int(cnn_network_config['pool_size'])))

    model.add(Dense(int(cnn_network_config["n_filters_dense"])))

    if cnn_network_config["batch_normalization"]:
        model.add(tf.keras.layers.BatchNormalization())

    model.add(Dropout(cnn_network_config['dropout']))

    model.add(Flatten())
    model.add(Dense(n_classes_out, activation='softmax', name='output1'))

    return model 

def get_cnn_lstm(input_shape, n_classes_out, cnn_lstm_network_config = None):
    '''
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9043535
    '''
    if type(cnn_lstm_network_config) == type(None):
        print("Loading default cnn lstm configuration.")
        cnn_lstm_network_config = CNN_LSTM_NETWORK_CONFIG

    n_timesteps = input_shape[0]
    n_features = input_shape[1]
    # '''
    # CNN-LSTM implementation
    # based on https://diglib.eg.org/handle/10.2312/egve20211326
    # LSTM expects shape (samples, time_stamps, num_features)
    # '''
    input_layer = tf.keras.layers.Input(shape = (n_timesteps, n_features))

    for i in range(cnn_lstm_network_config['conv_layers_number']):
        if i == 0:
            conv = tf.keras.layers.Conv1D(filters=cnn_lstm_network_config[f'n_filters_conv_{i}'], kernel_size=cnn_lstm_network_config['conv_filter_size'], padding="same")(input_layer)
        else: 
            conv = tf.keras.layers.Conv1D(filters=cnn_lstm_network_config[f'n_filters_conv_{i}'], kernel_size=cnn_lstm_network_config['conv_filter_size'], padding="same")(conv)
        if cnn_lstm_network_config["batch_normalization"]:
            conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.ReLU()(conv)

    max_pooling = tf.keras.layers.MaxPool1D(pool_size=int(cnn_lstm_network_config['pool_size']))(conv) # might change to 2D

    for i in range(cnn_lstm_network_config['lstm_layers_number']):
        if i == 0:
            lstm  = tf.keras.layers.LSTM(cnn_lstm_network_config[f'n_filters_lstm_{i}'], return_sequences=True, activation = "relu")(max_pooling)
        else: 
            lstm  = tf.keras.layers.LSTM(cnn_lstm_network_config[f'n_filters_lstm_{i}'], return_sequences=True, activation = "relu")(lstm)

    dropout = tf.keras.layers.Dropout(cnn_lstm_network_config['dropout'])(lstm)

    flatten = tf.keras.layers.Flatten()(dropout)

    for i in range(cnn_lstm_network_config['dense_layers_number']):
        if i == 0:
            dense = tf.keras.layers.Dense(cnn_lstm_network_config[f'n_filters_dense_{i}'], activation="relu")(flatten) 
        else: 
            dense = tf.keras.layers.Dense(cnn_lstm_network_config[f'n_filters_dense_{i}'], activation="relu")(dense) 
    
    dense_output = tf.keras.layers.Dense(n_classes_out, activation="softmax")(dense) 

    return tf.keras.models.Model(inputs=input_layer, outputs = dense_output)

def get_lstm(input_shape, n_classes_out, lstm_network_config = None):
    '''
    Standard LSTM implementation taken from 
    https://towardsdatascience.com/understanding-lstm-and-its-quick-implementation-in-keras-for-sentiment-analysis-af410fd85b47
    '''
    if type(lstm_network_config) == type(None):
        print("Loading default cnn lstm configuration.")
        lstm_network_config = LSTM_NETWORK_CONFIG
        
    n_timesteps = input_shape[0]
    n_features = input_shape[1]
    model = Sequential()   
        
    model.add(tf.keras.layers.Input(shape = (n_timesteps, n_features)))

    for i in range(lstm_network_config['lstm_layers_number']):
        model.add(LSTM(lstm_network_config[f'n_filters_lstm_{i}'], return_sequences=True)) #, activation = "relu"))
    
    if lstm_network_config["batch_normalization"]:
        model.add(tf.keras.layers.BatchNormalization())

    # https://machinelearningknowledge.ai/keras-lstm-layer-explained-for-beginners-with-example/ suggests dropout after every layer 
    model.add(Dropout(lstm_network_config['dropout']))

    model.add(Flatten())

    for i in range(lstm_network_config['dense_layers_number']):
        model.add(Dense(lstm_network_config[f'n_filters_dense_{i}'], activation="relu")) #, kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))) 

    model.add(Dense(n_classes_out, activation = "softmax"))

    return model


def get_lstm_simple(input_shape, n_classes_out, lstm_simple_network_config = None):
    '''
    A simple LSTM network
    '''
    # Get the config
    if type(lstm_simple_network_config) == type(None):
        print("Loading default simple lstm configuration.")
        lstm_simple_network_config = LSTM_SIMPLE_NETWORK_CONFIG

    # Get network parameters from config, for default parameters se "neural_networks_configurations.py"
    lstm_units =    lstm_simple_network_config["lstm_units"]
    dropout =       lstm_simple_network_config["dropout"]   
    # activation =    lstm_simple_network_config["activation"]
    
    # Simple lstm model structure with one lstm layer
    model = Sequential()
    model.add(LSTM(units = lstm_units, name='lstm_layer', input_shape = input_shape))
    model.add(Dropout(dropout, name = "dropout_layer"))
    model.add(Flatten(name='flatten_layer'))
    model.add(Dense(units=n_classes_out, activation="softmax", name='output'))

    return model

def get_cnn_simple(input_shape, n_classes_out, cnn_simple_network_config = None):
    '''
    Simple CNN network.
    Basically made up of either one or two CNN layers and 
    a Dense layer.

    input_shape = (n_timesteps, n_features)
    '''
    # Get the config
    if type(cnn_simple_network_config) == type(None):
        print("Loading default simple lstm configuration.")
        cnn_simple_network_config = CNN_SIMPLE_NETWORK_CONFIG

    # Get network parameters from config, for default parameters se "neural_networks_configurations.py"
    n_layers_conv       = cnn_simple_network_config["n_layers_conv"]
    n_filters_conv      = cnn_simple_network_config["n_filters_conv"]
    kernel_size_conv    = cnn_simple_network_config["kernel_size_conv"]
    dropout             = cnn_simple_network_config["dropout"] #default 0.5
    dense_units         = cnn_simple_network_config["dense_units"]
    pool_size           = cnn_simple_network_config["pool_size"]
    pool_type           = cnn_simple_network_config["pool_type"]  

    if input_shape[0] < 32: 
        n_layers_conv = 1

    if pool_type == "avg":
        pool_layer = AveragePooling1D(pool_size=pool_size)
    elif pool_type == "max":
        pool_layer = MaxPooling1D(pool_size=pool_size)
    else:
        print(f"Invalid pool type {pool_type}")
        sys.exit()

    model = Sequential()
    model.add(Conv1D(filters=n_filters_conv, kernel_size=kernel_size_conv, activation='relu', input_shape=input_shape, name = "conv_layer_1"))
    model.add(pool_layer)
    if n_layers_conv == 2: # currently only 1 or 2 layers possible
        model.add(Conv1D(filters=n_filters_conv, kernel_size=kernel_size_conv, activation='relu', name = "conv_layer_2"))
        model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Dropout(dropout, name = "dropout_layer"))
    model.add(Flatten(name = "flatten_layer"))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(n_classes_out, activation='softmax'))

    return model

def get_cnn_lstm_simple(input_shape, n_classes_out, cnn_lstm_simple_network_config = None):
    '''
    Simple CNN LSTM network.
    Basically made up of either one or two CNN layers and 
    a LSTM layer.
    '''
    # Get default config if None supplied
    if type(cnn_lstm_simple_network_config) == type(None):
        print("Loading default simple lstm configuration.")
        cnn_lstm_simple_network_config = CNN_LSTM_SIMPLE_NETOWRK_CONFIG
    
    if input_shape[0] < 32: 
        n_layers_conv = 1

    # Get network parameters from config, for default parameters se "neural_networks_configurations.py"
    n_layers_conv       = cnn_lstm_simple_network_config["n_layers_conv"]
    n_filters_conv      = cnn_lstm_simple_network_config["n_filters_conv"]
    kernel_size_conv    = cnn_lstm_simple_network_config["kernel_size_conv"]
    lstm_units          = cnn_lstm_simple_network_config["lstm_units"]
    conv_dropout        = cnn_lstm_simple_network_config["conv_dropout"]   
    lstm_dropout        = cnn_lstm_simple_network_config["lstm_dropout"]   
    pool_size           = cnn_lstm_simple_network_config["pool_size"]
    pool_type           = cnn_lstm_simple_network_config["pool_type"]     
    # activation          = cnn_lstm_simple_network_config["activation"]

    if pool_type == "avg":
        pool_layer = AveragePooling1D(pool_size=pool_size)
    elif pool_type == "max":
        pool_layer = MaxPooling1D(pool_size=pool_size)
    else:
        print(f"Invalid pool type {pool_type}")
        sys.exit()

    # Define model structure
    model = Sequential()

    model.add(Conv1D(filters=n_filters_conv, kernel_size=kernel_size_conv, activation='relu', input_shape=input_shape, name = "conv_layer_1"))
    model.add(pool_layer)
    if n_layers_conv == 2: # currently only 1 or 2 layers possible
        model.add(Conv1D(filters=n_filters_conv, kernel_size=kernel_size_conv, activation='relu', name = "conv_layer_2"))
        model.add(pool_layer)
    model.add(Dropout(conv_dropout, name = "conv_dropout_layer"))
    model.add(LSTM(lstm_units, name = "lstm_layer"))
    model.add(Dropout(lstm_dropout, name = "lstm_dropout_layer"))
    model.add(Flatten(name = "flatten_layer"))
    model.add(Dense(n_classes_out, activation="softmax", name = "fc_layer"))

    return model
