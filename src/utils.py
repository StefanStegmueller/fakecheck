#!/usr/bin/python

import os
import datetime
import tensorflow as tf
from tensorflow.data import Dataset

def batch_predict(tf_ds, batch_size, prediction_func):
    """Returns list of tuples (article_id, prediction, label).
    
    Predicts tensorflow dataset in batches.
    """
    
    evaluation_data = []
    for aids, inps, lbls in tf_ds.batch(batch_size).as_numpy_iterator():
        ps = prediction_func(inps)
        evaluation_data += zip(aids, ps, lbls)
    return evaluation_data

def chunks(lst, n):    
    """Yield successive n-sized chunks from lst."""
    
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
def get_checkpoint_callback(model_path, monitor_value, weights_only = False):
    return tf.keras.callbacks.ModelCheckpoint(model_path, 
                                              save_weights_only=weights_only,
                                              monitor=monitor_value,
                                              verbose=1, 
                                              save_best_only=True,
                                              mode='max')

def get_tensorboard_callback(log_dir_name):
    log_dir = os.path.join(log_dir_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    return tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)