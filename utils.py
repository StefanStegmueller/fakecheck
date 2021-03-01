#!/usr/bin/python

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
        