#!/usr/bin/python

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def format_germanfc(raw_data, spacy_model):
    """Returns dict with the following keys and the max. length of a sentence:
    'article_id', 'org', 'lbl', 'tokenized', 'tokenized_lower'.
    
    A sentence with 'lbl'=1.0 is considered a fake statement.
    """
    
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

    data = []
    max_sent_len = 0
    for article_id, article in enumerate(raw_data):
        title = spacy_model(article['Title']).sents
        teaser = spacy_model(article['Teaser']).sents
        text = spacy_model(article['Text']).sents

        p_title, max_sent_len = process_sentences(title, article_id, max_sent_len)
        p_teaser, max_sent_len = process_sentences(teaser, article_id, max_sent_len)
        p_text, max_sent_len = process_sentences(text, article_id, max_sent_len)

        article_data = p_title + p_teaser + p_text

        # Label sentences
        false_statements = [article['False_Statement_1'], article['False_Statement_2'], article['False_Statement_3']]     
        for fs in false_statements:
            if fs != '':
                fs_tokens = [t.text.lower() for t in spacy_model(fs)]
                matches = [count_matches(fs_tokens, t) for t in [d['tokenized_lower'] for d in article_data]]
                m = max(matches)
                max_indexes = [i for i, j in enumerate(matches) if j == m]

                for mi in max_indexes:
                    article_data[mi]['lbl'] = 1.0

        data = data + article_data
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

def contrastive_sampling(data, w2v_model, k):
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

    # only use train data
    # no negative sampling for test data neccesary
    sentence_embeddings = list(compute_sentence_embeddings(data))

    similarities = cosine_similarity(sentence_embeddings, sentence_embeddings)

    processed_topk_candidates = []
    ixs_topk_candidates = []
    for i, row_sims in enumerate(similarities):
        top_k_ixs = retrieve_topk_ixs(i, data, k, row_sims)

        top_k_processed = []    
        for top_k_ix in top_k_ixs:
            top_k_processed.append(data[top_k_ix]['processed']) 
        processed_topk_candidates.append(top_k_processed)
        ixs_topk_candidates.append(top_k_ixs)


    def assign_candidate(d, ptc, ix):
        d_copy = dict(d)
        d_copy['cs'] = ptc
        d_copy['cs_ix'] = ix
        return d_copy

    candidates_zipped = zip(data, processed_topk_candidates, ixs_topk_candidates)
    data = [[assign_candidate(d, ptc, ix) for ptc, ix in zip(ptcs, ixs)] for d, ptcs, ixs in candidates_zipped]

    flatten = lambda lst: [j for sub in lst for j in sub]
    return flatten(data)


def process_bert(data, max_sent_len, bert_tokenizer):
    sentences = [d['org'] for d in data]
    encodings =  bert_tokenizer(sentences, max_length=max_sent_len, truncation=True, padding=True)
    dl2ld = lambda dictionary: [dict(zip(dictionary,t)) for t in zip(*dictionary.values())] 
    encodings = dl2ld(encodings) 

    for i, encoding in enumerate(encodings):
        data[i]['input_ids'] = encoding['input_ids']
        data[i]['token_type_ids'] = encoding['token_type_ids']
        data[i]['attention_mask'] = encoding['attention_mask']        
    
    return data