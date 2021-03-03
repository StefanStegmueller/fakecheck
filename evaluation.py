#!/usr/bin/python

def __count_articles(data):
    return len(set([ida for ida, _, _ in data]))

def __map_articles(data, func):
    article_ids = set([ida for ida, _, _ in data])
    
    for id_article in article_ids:
        article_examples = [(p,t) for ida, p, t in data if ida == id_article]
        ps = [p for p,t in article_examples]
        ts = [t for p,t in article_examples]
        
        func(ps, ts)
        
def __ranked_indices(ps):
    return [i for i, v in sorted(enumerate(ps), key=lambda tup: tup[1], reverse=True)]

def precision_at_k(data, k):
    """Returns a float
    
    Input format for data is list of tuples (article_id, prediction, label).
    """
    
    precisions_at_k = []
    num_articles = __count_articles(data)
    
    def compute_precision_at_k(ps, ts):
        ranked_indices = __ranked_indices(ps)[:k]
        
        num_correct = 0
        for i, line_number in enumerate(ranked_indices):
            if ts[line_number] == 1:
                num_correct += 1
        precision = (num_correct / k)
        precisions_at_k.append(precision)
    
    __map_articles(data, compute_precision_at_k)
    
    return sum(precisions_at_k) / num_articles
    

def mean_average_precision(data):
    """Returns a float
    
    Input format for data is list of tuples (article_id, prediction, label).

    MAP metric is based on the official CLEF2019 implementation: 
    https://github.com/apepa/clef2019-factchecking-task1/blob/7d463336897ad1f870cb6a481953b94550c788a7/scorer/main.py#L52
    """
    
    avg_precisions = []
    num_articles = __count_articles(data)
    
    def compute_avg_precision(ps, ts):
        num_positive = sum(ts)
        ranked_indices = __ranked_indices(ps)
        
        precisions = []
        num_correct = 0
        for i, line_number in enumerate(ranked_indices):
            if ts[line_number] == 1:
                num_correct += 1
                precisions.append(num_correct / (i + 1))
            
        if precisions:
            avg_precisions.append(sum(precisions) / num_positive)
        else:
            avg_precisions.append(0)
    
    __map_articles(data, compute_avg_precision)
        
    return sum(avg_precisions) / num_articles 