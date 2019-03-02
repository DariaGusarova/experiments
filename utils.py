from copy import deepcopy
from sklearn.model_selection import train_test_split
import numpy as np
from IPython.display import clear_output



def read_queries_crash(path):
    f = open(path, 'r')
    queries = []
    
    for line in f:
        queries.append(line.split('\t')[1][5:].lower().split())
    return queries

def read_queries(path):
    f = open(path, 'r')
    queries = []
    
    for line in f:
        if len(line.split()) > 0:
            queries.append(line.lower().split())
    return queries


def split_to_train_and_validation(data, valid_size=0.2):
    """
    split input requests into train and validation parts
    """
    train, valid = train_test_split(data, test_size=valid_size)
    return train, valid


def transform(data):
    """
    transforms input requests to pairs of prefix and next word
    """
    prefixs = []
    next_words = []
    
    for sent in data:
        pref = []
        pref.append(sent[0])
        for i in range(1, len(sent)):
            prefixs.append(deepcopy(pref))
            next_words.append(sent[i])
            pref.append(sent[i])
            
    return prefixs, next_words


def as_matrix(emb, emb_size, data, max_len=None):    
    max_len = min(max(map(len, data)), max_len or float('inf'))
    
    unk = np.zeros(emb_size)
    matrix = np.full((len(data), max_len+1, emb_size), unk)
    
    for i, seq in enumerate(data):
        seq_matrix = []
        for word in seq:
            if word in emb.vocab:
                seq_matrix.append(emb[word])
            else:
                seq_matrix.append(unk)
        matrix[i, -len(seq):] = seq_matrix
    
    return matrix

def as_matrix_target(emb, emb_size, data, max_len=None):    
    unk = np.zeros(emb_size)
    matrix = np.full((len(data), emb_size), unk)
    
    for i, word in enumerate(data):
        if word in emb.vocab:
            matrix[i] = emb[word]
  
    return matrix


def iterate_minibatches(emb, emb_size, prefixs, next_words, batch_size=256, shuffle=True, cycle=False):
    """ iterates minibatches of data in random order """
    
    while True:
        indices = np.arange(len(prefixs))
        if shuffle:
            indices = np.random.permutation(indices)

        for start in range(0, len(indices), batch_size):
            batch = as_matrix(emb, emb_size, prefixs[indices[start : start + batch_size]])
            target = as_matrix_target(emb, emb_size, next_words[indices[start : start + batch_size]])
            yield batch, target
        
        if not cycle: break
            
def calculate_accuracy(model, emb, emb_size, batch_size, test_prefixs, test_next_words):  
    all_accr = 0
    all_count = 0
    for batch_x, batch_y in iterate_minibatches(emb, emb_size, test_prefixs, test_next_words, batch_size=batch_size, shuffle=False):
        batch_pred = model.predict(batch_x)
        for j, k in enumerate(batch_pred):
            if emb.most_similar([k], topn=1)[0][0] == emb.most_similar([batch_y[j]], topn=1)[0][0]:
                all_accr += 1
            all_count += 1
        clear_output()
        print('In progress {} / {}, accuracy = {}'.format(all_count, len(test_prefixs), all_accr / all_count), flush=True)
     
    return all_accr / all_count



