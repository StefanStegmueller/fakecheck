#!/usr/bin/python

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def __process_article(article_id, article, statement_desc, spacy_model, max_sent_len):    
    def count_matches(false_statement, sentence):
        count = 0
        sent_copy = sentence[:]
        for w in false_statement:
            if w in sent_copy:
                count += 1
                sent_copy.remove(w)
        return count
    
    def process_sentences(sentences, article_id,  max_sent_len):
        processed = []
        for s in sentences:
            # ignore sentences of length 1
            if len(s) <= 1:
                continue
            # ignore sentences consisting exclusively of punctuation
            if not any([not t.is_punct for t in s]):
                continue
            # ignore sentences not containing any letter
            if not any([any([c.isalpha() for c in t.text]) for t in s]):
                continue
            if len(s) > max_sent_len:
                max_sent_len = len(s)
            processed.append({
                'article_id': article_id,
                'org': s.text,
                'lbl': 0.0,
                'tokenized': [t.text for t in s],
                'tokenized_lower': [t.text.lower() for t in s]
            })
        return processed, max_sent_len
    
    article_data = []
    
    title = spacy_model(article['Title']).sents
    p_title, max_sent_len = process_sentences(title, article_id, max_sent_len)
    article_data = article_data + p_title
    
    if 'Teaser' in article:
        teaser = spacy_model(article['Teaser']).sents        
        p_teaser, max_sent_len = process_sentences(teaser, article_id, max_sent_len)
        article_data = article_data + p_teaser
        
    text = spacy_model(article['Text']).sents
    p_text, max_sent_len = process_sentences(text, article_id, max_sent_len)
    article_data = article_data + p_text

    # Label sentences
    statements = [] 
    
    def add_statement_safe(ix_statement):
        full_statement_desc = statement_desc + '_Statement_' + ix_statement 
        if full_statement_desc in article:
            statements.append((int(ix_statement), article[full_statement_desc]))
    
    add_statement_safe('1')
    add_statement_safe('2')
    add_statement_safe('3')
     
    for i, s in statements:
        if s != '':
            s_tokens = [t.text.lower() for t in spacy_model(s)]
            matches = [count_matches(s_tokens, t) for t in 
                       [d['tokenized_lower'] for d in article_data]]
            max_match = max(matches)
            max_indexes = [i for i, j in enumerate(matches) if j == max_match]

            for mi in max_indexes:
                article_data[mi]['lbl'] = 1.0
                article_data[mi]['statement_id'] = i
    
    return article_data, max_sent_len


def format_germanfc(raw_data, spacy_model, max_sent_len):
    """Returns dict with the following keys and the max. length of sentences:
    'article_id', 'org', 'lbl', 'tokenized', 'tokenized_lower'.
    
    A sentence with 'lbl'=1.0 is considered a fake statement.
    """
    data = []
    for article_id, article in enumerate(raw_data):        
        article_data, max_sent_len = __process_article(article_id, article, 'False', spacy_model, max_sent_len)

        data = data + article_data
    return data, max_sent_len


def format_germantrc(raw_data, spacy_model, max_sent_len):
    """Returns dict with the following keys and the max. length of sentences:
    'article_id', 'org', 'lbl', 'tokenized', 'tokenized_lower'.
    
    A sentence with 'lbl'=1.0 is considered a true statement.
    """
    
    data = []
    for id_raw in raw_data.keys():
        aid = int(id_raw.replace("ID = ", ""))
        true_news = raw_data[id_raw][1]
    
        for tn in true_news:
            article_data, max_sent_len = __process_article(aid, tn, 'True', spacy_model, max_sent_len)
            data = data + (article_data)
    return data, max_sent_len


def split_dataset(data, split_factor):
    """Separates data while not separating data from different articles."""
    
    num_articles = len(set([d['article_id'] for d in data]))
    
    num_train_articles = int(split_factor * num_articles)
    data_split1 = list(filter(lambda d: d['article_id'] <= num_train_articles, data))
    data_split2 = list(filter(lambda d: d['article_id'] > num_train_articles, data))
    
    return data_split1, data_split2


def process_hansen(data, max_sent_len, w2v_model, spacy_model):
    def to_deps(doc, max_sent_len):
        oh_vectors = []
        for token in doc:
            vec = np.zeros(max_sent_len)
            vec[token.head.i] = 1
            oh_vectors.append(vec)

        # padding with 0 vectors to max sentence length
        while len(oh_vectors) < max_sent_len:
            oh_vectors.append(np.zeros(max_sent_len))
        return oh_vectors

    def embed(sentence, max_sent_len):
        vectorized_sentence = []
        vector_dim = w2v_model.wv.vector_size
        for word in sentence:
            if word in w2v_model.wv:
                vectorized_sentence.append(w2v_model.wv[word])
            else:
                vectorized_sentence.append(np.zeros(vector_dim))

        # padding with 0 vectors to max sentence length
        while len(vectorized_sentence) < max_sent_len:
            vectorized_sentence.append(np.zeros(vector_dim))

        return vectorized_sentence
    
    for d in data:
        doc = spacy_model(d['org'])
        dep_vectors = to_deps(doc, max_sent_len)
        embedded_words = embed(d['tokenized_lower'], max_sent_len)
        d['processed'] = np.concatenate((embedded_words, dep_vectors), axis=1)
        
    return data

def contrastive_sampling(data, w2v_model, k, true_data=None, assign_bert=False):
    def compute_sentence_embeddings(data):
        word_vector_dim = w2v_model.wv.vector_size
        for d in data:
            word_embeddings = [w[:word_vector_dim] for w in d['processed']]
            yield np.mean(word_embeddings, axis=0)

    def retrieve_topk_ixs(entry_index, data, k, sims):
        topk_stack = [(0,0)]

        for i, sim in enumerate(sims):
            is_greater = any([sim > tk_sim for (index, tk_sim) in topk_stack])
            negative_label = data[entry_index]['lbl'] != data[i]['lbl']
            not_own_sim = entry_index != i

            if is_greater and negative_label and not_own_sim: 
                if len(topk_stack) >= k:
                    topk_stack.pop()

                topk_stack.append((i, sim))    
                topk_stack.sort(reverse=True)
        return [index for (index, sim) in topk_stack]
    
    def retrieve_processed_data(data, index):
        if assign_bert:
            return (data[index]['input_ids'],
                    data[index]['token_type_ids'],
                    data[index]['attention_mask'])
        else:
            return data[index]['processed']
        
    def assign_candidate(d, ptc):
        d_copy = dict(d)
        if assign_bert:
            inp_ix_c, token_ids_c, att_mask_c = ptc
            
            # rename field dict fields
            d_copy['input_ids1'] = d_copy.pop('input_ids')
            d_copy['token_type_ids1'] = d_copy.pop('token_type_ids')
            d_copy['attention_mask1'] = d_copy.pop('attention_mask')

            d_copy['input_ids2'] = inp_ix_c
            d_copy['token_type_ids2'] = token_ids_c
            d_copy['attention_mask2'] = att_mask_c
        else:
            d_copy['cs'] = ptc
        return d_copy

        
    sentence_embeddings = list(compute_sentence_embeddings(data))
    similarities = cosine_similarity(sentence_embeddings, sentence_embeddings)

    processed_topk_candidates = []
    ixs_topk_candidates = []
    for i, row_sims in enumerate(similarities):
        top_k_ixs = retrieve_topk_ixs(i, data, k, row_sims)
        top_k_processed = []
        
        # add true statements as contrastive samples
        if data[i]['lbl' ] == 1.0 and true_data:
            aid = data[i]['article_id']
            sid = data[i]['statement_id']
            true_statement_ids = [i for i, d in enumerate(true_data)
                               if d['article_id'] == aid 
                               and d['lbl'] == 1.0
                               and d['statement_id'] == sid]
            for tsi in true_statement_ids:
                if top_k_ixs:
                    top_k_ixs.pop()
                top_k_processed.append(retrieve_processed_data(true_data, tsi))
            
        for top_k_ix in top_k_ixs:
            top_k_processed.append(retrieve_processed_data(data, top_k_ix))
        processed_topk_candidates.append(top_k_processed)

    candidates_zipped = zip(data, processed_topk_candidates)
    data = [[assign_candidate(d, ptc) for ptc in ptcs] for d, ptcs in candidates_zipped]

    flatten = lambda lst: [j for sub in lst for j in sub]
    return flatten(data)


def process_bert(data, max_sent_len, bert_tokenizer):
    sentences = [d['org'] for d in data]
    encodings =  bert_tokenizer(sentences, max_length=max_sent_len, truncation=True, padding='max_length')
    dl2ld = lambda dictionary: [dict(zip(dictionary,t)) for t in zip(*dictionary.values())] 
    encodings = dl2ld(encodings) 

    for i, encoding in enumerate(encodings):
        data[i]['input_ids'] = encoding['input_ids']
        data[i]['token_type_ids'] = encoding['token_type_ids']
        data[i]['attention_mask'] = encoding['attention_mask']        
    
    return data