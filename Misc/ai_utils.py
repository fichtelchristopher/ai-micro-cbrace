import tensorflow as tf
import pandas as pd 
import numpy as np
import sys

from keras import regularizers

import math
import tensorflow as tf
tf.random.set_seed(7)

from Configs.pipeline_config import *
from Configs.namespaces import *
from Configs.nn_pipeline_config import *

from Misc.dataset_utils import *

from keras.callbacks import ModelCheckpoint, EarlyStopping

from time import process_time

from Models.neural_networks import * 

from keras.models import Model

from sklearn.metrics import f1_score, accuracy_score

# class EarlyStopping(tf.keras.callbacks.Callback):
#   def __init__(self, accuracy_threshold):
#     super().__init__()
#     self.accuracy_treshold = accuracy_threshold

#   def on_epoch_end(self, epoch, logs={}):
#     if(logs.get('accuracy') > self.accuracy_treshold):
#       print("\nAccuracy threshold reached. Training stopped.")
#       self.model.stop_training = True

class TrainingCancellationCallback(tf.keras.callbacks.Callback):
  def __init__(self, f1_threshold, n_epochs):
    super().__init__()
    self.f1_threshold = f1_threshold
    self.n_epochs = n_epochs
  
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('val_f1') < self.f1_threshold):
      if epoch > 20:
        print("\nTraining loss exceeds threshold. Training cancelled.")
        self.model.stop_training = True
    
    if(logs.get('val_f1') < self.f1_threshold - 0.1):
      if epoch > 10:
        print("\nTraining loss exceeds threshold. Training cancelled.")
        self.model.stop_training = True

    # if (logs.get("val_f1")) < self.f1_treshold:
    #   if epoch > 20: 
    #     print(f"\nTraining f1 scores below threshold {self.f1_treshold} after more than 10 epochs. Training cancelled.")
    #     self.model.stop_training = True

    # if (logs.get("f1")) < self.f1_treshold + 0.1:
    #   if epoch > 20: 
    #     print(f"\nTraining f1 scores below threshold {self.f1_treshold + 0.1} after more than 20 epochs. Training cancelled.")
    #     self.model.stop_training = True

    # if np.isnan(logs.get("loss")):
    #   print("Training loss is nan.")
    #   self.model.stop_training = True

class TrainingCallback(tf.keras.callbacks.Callback):
  def __init__(self, model_output_path):
    '''
    :param lr_file: save the learning rate over time.
    '''
    self.model_output_path = model_output_path
    super().__init__()

  def on_epoch_end(self, epoch, logs=None):
    if epoch > 0:
      model_history = self.model.history.history
      save_history_plots(model_history, self.model_output_path)

    return super().on_epoch_end(epoch, logs)

  def on_train_begin(self, logs=None):
    return super().on_train_begin(logs)

  def on_train_end(self, logs=None):
    return super().on_train_end(logs)

  def on_epoch_begin(self, epoch, logs=None):
    print(f"LR at epoch {epoch} is " + str(self.model.optimizer.lr.numpy()))
    return super().on_epoch_begin(epoch, logs)
  
def ce_loss(y_true, y_pred):
  log_y_pred = tf.math.log(y_pred + sys.float_info.epsilon)
  elements = -tf.math.multiply_no_nan(x = log_y_pred, y = y_true)
  
  ce_loss_val = tf.reduce_mean(tf.reduce_sum(elements, axis = 1))

  '''
  if math.isinf(ce_loss_val):
    print("\n\n\nCE loss is infinity.")
    print("True labels")
    print(y_true)
    print("\nPredicted labels")
    print(y_pred)
  
  if math.isnan(ce_loss_val):
    print("\n\n\nCE loss is nan.")
    print("True labels")
    print(y_true)
    print("\nPredicted labels")
    print(y_pred)
    print("\n\n\n\n")
  '''

  return ce_loss_val

def predict(model , X_val): 
  '''
  Predict the class labels for the data in val_X

  :param model: the (fitted) sklearn model
  :param val_X: data to be predicted (val or test )
  '''
  
  prediction = model.predict(X_val)

  return prediction


def predict_probability(model, X_data):
  '''
  If the classifier was created with the optional "probability = True" you can call this method 
  to calculate the probability per class 
  
  :return: df with integers (class names) as column names 
  '''
  prediction_probability = model.predict_proba(X_data)

  return pd.DataFrame(data = prediction_probability, columns=(model.classes_).astype("int32")) 


def f1(y_true, y_pred, training = True): #, labels = None):
  '''
  If training is True, y_true and y_pred are one hot encoded
  meaning shape (num_samples, num_classes) --> they have to be converted
  to a not one hot encoded shape of (num_samples, ) in order to compute
  the f1 score
  '''
  if training:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
  
    y_true = np.argmax(y_true, axis = 1)
    y_pred = np.argmax(y_pred, axis = 1)
    f1_score_macro = f1_score(y_true, y_pred, average='macro', zero_division='warn')

  return f1_score_macro

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
      return optimizer.lr
    return lr
  
def get_y_shape_metric():
  def y_type(y_true, y_pred):
    return 0
  return y_type 

def get_compiled_model(input_shape, n_classes_out, model_type, pipeline_configuration, network_configuration = None, use_val_data = False):
  '''
  :param input_shape: (n_timesteps, n_signals)
  :param n_classes_out: int
  :param model_type: str, one of the implementations defined in the neural network config
  :param network_configuration: if None --> select default configuration
  '''
  loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
  # loss = "categorical_crossentropy"
  optimizer = tf.keras.optimizers.Adam(learning_rate=pipeline_configuration["LR"])
  lr_metric = get_lr_metric(optimizer)
  
  metrics = ["accuracy", f1]

  if model_type == "LSTM":
      model = get_lstm(input_shape = input_shape, n_classes_out = n_classes_out, lstm_network_config=network_configuration)
  elif model_type == "CNN":
      model = get_cnn(input_shape = input_shape, n_classes_out = n_classes_out, cnn_network_config=network_configuration)
  elif model_type == "CNN_LSTM":
      model = get_cnn_lstm(input_shape = input_shape, n_classes_out = n_classes_out, cnn_lstm_network_config=network_configuration)
  elif model_type == "LSTM_SIMPLE":
      model = get_lstm_simple(input_shape = input_shape, n_classes_out = n_classes_out, lstm_simple_network_config=network_configuration)
  elif model_type == "CNN_SIMPLE":
      model = get_cnn_simple(input_shape = input_shape, n_classes_out = n_classes_out, cnn_simple_network_config=network_configuration)
  elif model_type == "CNN_LSTM_SIMPLE":
      model = get_cnn_lstm_simple(input_shape = input_shape, n_classes_out = n_classes_out, cnn_lstm_simple_network_config=network_configuration)
  else:
      print(f"Unknown neural network architecture {model_type}.")

  model.compile(loss=loss, 
                  optimizer=optimizer,
                  metrics=metrics,
                  run_eagerly = True)
                  # loss_weights=regularizers.l2(pipeline_configuration["REGULARIZATION_STRENGTH"]))
  return model

def train_model(model, train_set, epochs, pipeline_configuration, model_output_path, activity_code_type_dict, val_set = None, early_stopping_acc_threshold = 0.99, loss_treshold = 5.0, f1_threshold = 0.45, class_weight_dict = None): 
    
    '''
    Create a classificator, train and predict
    '''
    if type(class_weight_dict) == type(None):
      class_weight_dict = {}
      for _code, _type in activity_code_type_dict.items():
        class_weight_dict[_code] = 1
    class_weight_dict = encode_classweight_dict(class_weight_dict, activity_dict = activity_code_type_dict)

    # Callback for stopping training when accuracy exceeds a threshold
    # early_stopping_acc_threshold_callback = EarlyStopping(early_stopping_acc_threshold)
    early_stopping_callback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

    # Callback for cancelling training when loss exceeds a threshold or when f1 score is below a threshold after a certain amount of epochs
    n_epochs = 20
    training_cancellation_callback = TrainingCancellationCallback(f1_threshold, n_epochs)

    # Callback to monitor training process
    training_callback = TrainingCallback(model_output_path)

    # save best model checkpoint
    monitor = "loss" if type(val_set) == type(None) else "val_loss"
    model_checkpoint_callback = ModelCheckpoint(filepath=model_output_path, 
                             monitor=monitor,
                             verbose=1, 
                             save_best_only=True,
                             mode="min")

    # Learning Rate Scheduler
    def learning_rate_schedule(epoch, lr):
      if (epoch % pipeline_configuration["LR_DECAY_AFTER_EPOCHS"] == 0) & (epoch != 0):
        return lr * pipeline_configuration["LR_DECAY"]
      else:
        return lr
    learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(learning_rate_schedule)

    t1_start = process_time()
    print(f"Fitting model for {epochs} epochs.")
    callbacks = [learning_rate_callback, model_checkpoint_callback, training_callback, early_stopping_callback]#, training_cancellation_callback] #, training_callback, training_cancellation_callback
    # history = model.fit(train_set,  class_weight = class_weight_dict, verbose = 1)
    history = model.fit(train_set, validation_data = val_set, callbacks = callbacks, epochs=epochs, verbose = 2)
    t1_stop = process_time() 
    print("Fitting model time: ", t1_stop-t1_start, "s")

    return model

def save_history_plots(model_history, classifier_output_path):
  '''
  :param model_history: a trained models's .history.history object
  :param classifier_output_path: the path to where the trained model is saved, in h5 format --> has to end on h5
  '''
  loss = model_history["loss"]
  f1_score = model_history["f1"] 
  accuracy = model_history["accuracy"]

  loss_plot_f = classifier_output_path.replace(".h5", "_loss.png")
  f1_score_f = classifier_output_path.replace(".h5", "_f1.png")
  accuracy_f = classifier_output_path.replace(".h5", "_acc.png")

  plt.plot(loss, color = METRIC_COLOR_DICT["LOSS"], label = "loss")
  plt.xlabel("epoch")
  plt.ylabel("cross entropy loss")
  if "val_loss" in list(model_history.keys()):
    plt.plot(model_history["val_loss"], color = METRIC_COLOR_DICT["VAL_LOSS"], label = "val_loss")
    plt.legend()
  plt.tight_layout()
  plt.savefig(loss_plot_f)
  plt.clf() 

  plt.plot(f1_score, color = METRIC_COLOR_DICT["F1_SCORE"], label = "f1_score")
  plt.xlabel("epoch")
  plt.ylabel("f1 score (multiclass)")
  if "val_f1" in list(model_history.keys()):
    plt.plot(model_history["val_f1"], color = METRIC_COLOR_DICT["VAL_F1_SCORE"], label = "val_f1_scores")
    plt.legend()
  plt.tight_layout()
  plt.savefig(f1_score_f)
  plt.clf() 

  plt.plot(accuracy, color = METRIC_COLOR_DICT["ACCURACY"], label = "train_accuracy")
  plt.xlabel("epoch")
  plt.ylabel("accuracy")
  if "val_accuracy" in list(model_history.keys()):
    plt.plot(model_history["val_accuracy"], color = METRIC_COLOR_DICT["VAL_ACCURACY"], label = "val_accuracy")
    plt.legend()
  plt.tight_layout()
  plt.savefig(accuracy_f)
  plt.clf() 

  if "lr" in model_history.keys():
    lr = model_history["lr"] 
    lr_f = classifier_output_path.replace(".h5", "_lr.png")  
    plt.plot(lr, color = METRIC_COLOR_DICT["LR"])
    plt.xlabel("epoch")
    plt.ylabel("learning rate")
    plt.tight_layout()
    plt.savefig(lr_f)
    plt.clf() 
