from __future__ import print_function, division
from builtins import range
# you may need to upudate your version of future
# with sudo pip install -U future

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
# pip install -U gensim
from gensim.models import KeyedVectors

# data from https://www.cs.umb.edu/~smimarog/textmining/datasets/
train = pd.read_csv('/Users/koheimoroi/Git/NLP/large_files/r8_train_all_terms.txt',
                    header=None, sep='\t')
test = pd.read_csv('/Users/koheimoroi/Git/NLP/large_files/r8_train_all_terms.txt',
                   header=None, sep='\t')
train.columns = ['label', 'content']
test.columns = ['label', 'content']


class GloveVectorizer:

    def __init__(self):
        # load in pre-trained word vectors
        print('loading word vertors...')
        word2vec = {}
        embedding = []
        idx2word = []
        with open('/Users/koheimoroi/Git/NLP/large_files/glove.6B/glove.6B.50d.txt') as f:
            # is just a space-separated text file in the format:
            # word vec[0] vec[1] vec[2] ...
            for line in f:
                values = line.split()
                if len(values) > 0:
                    word = values[0]
                    vec = np.asarray(values[1:], dtype='float32')
                    word2vec[word] = vec
                    embedding.append(vec)
                    idx2word.append(word)
            print('Found %s word vectors.' % len(word2vec))
        # save for later
        self.word2vec = word2vec
        self.embedding = np.array(embedding)
        self.word2idx = {v: k for k, v in enumerate(idx2word)}
        self.V, self.D = self.embedding.shape

    def fit(self, data):
        pass

    def transform(self, data):
        X = np.zeros((len(data), self.D))
        n = 0
        emptycount = 0
        for sentence in data:
            tokens = sentence.lower().split()
            vecs = []
            for word in tokens:
                if word in self.word2vec:
                    vec = self.word2vec[word]
                    vecs.append(vec)
            if len(vecs) > 0:
                vecs = np.array(vecs)
                X[n] = vecs.mean(axis=0)
            else:
                emptycount += 1
            n += 1
        print("Numer of samples with no words found: %s / %s" %
              (emptycount, len(data)))
        return X

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


if __name__ == '__main__':

    vectorizer = GloveVectorizer()
    # vectorizer = Word2VecVectorizer()
    Xtrain = vectorizer.fit_transform(train.content)
    Ytrain = train.label

    Xtest = vectorizer.transform(test.content)
    Ytest = test.label

    # create the model, train it, print scores
    model = RandomForestClassifier(n_estimators=200)
    model.fit(Xtrain, Ytrain)
    print("train score:", model.score(Xtrain, Ytrain))
    print("test score:", model.score(Xtest, Ytest))
