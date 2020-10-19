from collections import defaultdict

import numpy as np
import pickle

from scipy.stats import spearmanr
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import norm

import time

with open('./data/vocab-15kws.txt') as f:
    V = [l.strip() for l in f.readlines()]

with open('./data/vocab-5k.txt') as f:
    V_c = [l.strip() for l in f.readlines()]

l = 997898 # len(data)
def get_vectors_and_idf(V, V_c, windows=[1, 3, 6]):
    """
        txt: a list of sentence strings
        V: a list of vocabulary words
        V_c: a list of context words
        window: an integer, window length
    """
    idf = {}, {}
    word_to_ind = dict(zip(V, range(len(V))))
    contextword_to_ind = dict(zip(V_c, range(len(V_c))))
    df_v, df_v_c = dok_matrix((len(V), 1), dtype=np.float32), dok_matrix((len(V_c), 1), dtype=np.float32)
    t1 = time.time()
    vectors = {k: dok_matrix((len(V), len(V)), dtype=np.float32) for k in [('V', 1), ('V', 3), ('V', 6)]}
    for k in [('V_c', 1), ('V_c', 3), ('V_c', 6)]:
        vectors[k] = dok_matrix((len(V), len(V_c)), dtype=np.float32)
    with open('./data/wiki-1percent.txt') as f:
        for linenum, line in enumerate(f):
            if (linenum+1)%50000==0 or linenum==0:
                t2 = time.time()
                print(f"({linenum+1}/{l}) : saving at {t2-t1} s")
                with open(f'./data/vectors.pkl', 'wb') as f:
                    pickle.dump(vectors, f)
                with open(f'./data/idf.pkl', 'wb') as f:
                    pickle.dump(idf, f)
            line = line.split()
            for word_ind, word in enumerate(line):
                if word not in V:
                    continue
                df_v[word_to_ind[word]] = df_v[word_to_ind[word]] + 1
                if word in V_c:
                    df_v_c[contextword_to_ind[word]] = df_v_c[contextword_to_ind[word]]+1
                for w in windows:
                    idflist = []
                    for cword_ind in range(word_ind-w, word_ind+w):
                        if cword_ind != word_ind and cword_ind <= len(line)-1 and cword_ind >= 0:
                            contextword = line[cword_ind]
                            if contextword in V_c:
                                i, j = word_to_ind[word], contextword_to_ind[contextword]
                                vectors['V_c', w][i, j] = vectors['V_c', w][i, j] + 1
                            if contextword in V:
                                i, j = word_to_ind[word], word_to_ind[contextword]
                                vectors['V', w][i, j] = vectors['V', w][i, j] + 1
    df_v, df_v_c = df_v.toarray().flatten(), df_v_c.toarray().flatten()
    idf['V'] = np.array([l/x if x!=0 else 0. for x in df_v])
    idf['V_c'] = np.array([l/x if x!=0 else 0. for x in df_v_c])
    return vectors, idf

from collections import defaultdict

from tqdm import tqdm
import numpy as np
import pickle

from scipy.stats import spearmanr
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import norm

import time

with open('./data/vocab-15kws.txt') as f:
    V = [l.strip() for l in f.readlines()]

with open('./data/vocab-5k.txt') as f:
    V_c = [l.strip() for l in f.readlines()]

l = 997898 # len(data)
def get_vectors_and_idf(V, V_c, windows=[1, 3, 6]):
    """
        txt: a list of sentence strings
        V: a list of vocabulary words
        V_c: a list of context words
        window: an integer, window length
    """
    idf = {}, {}
    word_to_ind = dict(zip(V, range(len(V))))
    contextword_to_ind = dict(zip(V_c, range(len(V_c))))
    df_v, df_v_c = dok_matrix((len(V), 1), dtype=np.float32), dok_matrix((len(V_c), 1), dtype=np.float32)
    t1 = time.time()
    vectors = {k: dok_matrix((len(V), len(V)), dtype=np.float32) for k in [('V', 1), ('V', 3), ('V', 6)]}
    for k in [('V_c', 1), ('V_c', 3), ('V_c', 6)]:
        vectors[k] = dok_matrix((len(V), len(V_c)), dtype=np.float32)
    with open('./data/wiki-1percent.txt') as f:
        for linenum, line in enumerate(f):
            if (linenum+1)%300000==0 or linenum==0:
                t2 = time.time()
                print(f"({linenum+1}/{l}) : saving at {t2-t1} s")
                with open(f'./data/vectors.pkl', 'wb') as f:
                    pickle.dump(vectors, f)
                with open(f'./data/idf.pkl', 'wb') as f:
                    pickle.dump(idf, f)
            line = line.split()
            for word_ind, word in enumerate(line):
                if word not in V:
                    continue
                df_v[word_to_ind[word]] = df_v[word_to_ind[word]] + 1
                if word in V_c:
                    df_v_c[contextword_to_ind[word]] = df_v_c[contextword_to_ind[word]]+1
                for w in windows:
                    idflist = []
                    for cword_ind in range(word_ind-w, word_ind+w):
                        if cword_ind != word_ind and cword_ind <= len(line)-1 and cword_ind >= 0:
                            contextword = line[cword_ind]
                            if contextword in V_c:
                                i, j = word_to_ind[word], contextword_to_ind[contextword]
                                vectors['V_c', w][i, j] = vectors['V_c', w][i, j] + 1
                            if contextword in V:
                                i, j = word_to_ind[word], word_to_ind[contextword]
                                vectors['V', w][i, j] = vectors['V', w][i, j] + 1
    df_v, df_v_c = df_v.toarray().flatten(), df_v_c.toarray().flatten()
    idf['V'] = np.array([l/x if x!=0 else 0. for x in df_v])
    idf['V_c'] = np.array([l/x if x!=0 else 0. for x in df_v_c])
    return vectors, idf

def get_idf(V, V_c, startind=0, lastind=np.inf):
    """
        txt: a list of sentence strings
        V: a list of vocabulary words
        V_c: a list of context words
        window: an integer, window length
    """
    idf = {}
    word_to_ind = dict(zip(V, range(len(V))))
    contextword_to_ind = dict(zip(V_c, range(len(V_c))))
    V, V_c = set(V), set(V_c)
    df_v, df_v_c = dok_matrix((len(V), 1), dtype=np.float32), dok_matrix((len(V_c), 1), dtype=np.float32)
    t1 = time.time()
    with open('./data/wiki-1percent.txt') as f:
        for linenum, line in tqdm(enumerate(f)):
            if linenum<startind:
                continue
            if (linenum+1)%50000==0 or linenum==0:
                t2 = time.time()
                print(f"({linenum+1}/{l}) : {t2-t1} s")
                #with open(f'/Users/aabir/Documents/uchicago/nlp/hw1/vectors2.pkl', 'wb') as f:
                #    pickle.dump(vectors, f)
                #with open(f'/Users/aabir/Documents/uchicago/nlp/hw1/idf2.pkl', 'wb') as f:
                #    pickle.dump(idf, f)
            line = line.split()
            for word_ind, word in enumerate(line):
                if word not in V:
                    continue
                df_v[word_to_ind[word]] = df_v[word_to_ind[word]] + 1
                if word in V_c:
                    df_v_c[contextword_to_ind[word]] = df_v_c[contextword_to_ind[word]]+1
                continue
            if lastind==linenum:
                return df_v, df_v_c
    #idf['V'] = np.array([l/x if x!=0 else 0. for x in df_v])
    #idf['V_c'] = np.array([l/x if x!=0 else 0. for x in df_v_c])
    return df_v, df_v_c


# started at 9:32 pm on Oct 14 2020
t1 = time.time()
print(f"run started at {t1}")

#vectors, idfs = get_vectors_and_idf(V, V_c, [1, 3, 6])

df = get_idf(V, V_c, 0, 500000)

with open('./df-nonsh-0to500000.pkl', 'wb') as f:
    pickle.dump(df, f)

#with open(f'./vectors+idf.pkl', 'wb') as f:
#    pickle.dump([vectors, idfs], f)

t2 = time.time()
print(f"run finished at {t2} ({t2-t1} s)")
