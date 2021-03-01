#!/usr/bin/python

def mean_average_precision(data):
    """Returns a float
    
    Input format for data is list of tuples (article_id, prediction, label).

    MAP metric is based on the official CLEF2019 implementation: 
    https://github.com/apepa/clef2019-factchecking-task1/blob/7d463336897ad1f870cb6a481953b94550c788a7/scorer/main.py#L52
    """
    
    avg_precisions = []
    article_ids = set([ida for ida, _, _ in data])
    num_articles = len(article_ids)
    
    for id_article in article_ids:
        article_examples = [(p,t) for ida, p, t in data if ida == id_article]
        ps = [p for p,t in article_examples]
        ts = [t for p,t in article_examples]
        
        num_positive = sum(ts)

        ranked_indices = [i for i, v in sorted(enumerate(ps), key=lambda tup: tup[1], reverse=True)]
        
        # ++++ DEBUG CODE - START +++ #
        #hits = []
        #for i in range(len(ranked_indices)):
        #   if ys[ranked_indices[i]] == 1:
        #        hits.append(1)
        #    else:
        #        hits.append(0)
        #print(hits)
        # ++++ DEBUG CODE - END   +++ #
        
        precisions = []
        num_correct = 0
        for i in range(len(ranked_indices)):
            if ts[ranked_indices[i]] == 1:
                num_correct += 1
                precisions.append(num_correct / (i + 1))
            
        if precisions:
            avg_precisions.append(sum(precisions) / num_positive)
        else:
            avg_precisions.append(0)
        
    return sum(avg_precisions) / num_articles 