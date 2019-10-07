from __future__ import print_function, division
from builtins import range
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
from sklearn.utils import shuffle
from datetime import datetime
# from util import find_analogies

from scipy.spatial.distance import cosine as cos_dist
from sklearn.metrics.pairwise import pairwise_distances
from glob import glob
import os
import sys
import string

sys.path.append(os.path.abspath('..'))
from rnn.corpus import get_sentences_with_word2idx_limit_vocab as get_brown


# unfortunately these work different ways
def remove_punctuation_2(s):
    return s.translate(None, string.punctuation)

def remove_punctuation_3(s):
    return s.translate(str.maketrans('','',string.punctuation))

if sys.version.startswith('2'):
    remove_punctuation = remove_punctuation_2
else:
    remove_punctuation = remove_punctuation_3

def get_wiki():
    V = 20000
    files = glob('../large_file/enwiki.txt')
    all_word_counts = {}
    for f in files:
        for line in open(f):
            if line and line[0] not in '[*-|=\{\}':
                s = remove_punctuation(line).lower().split()
                if len(s) > 1:
                    for word in s:
                        if word not in all_word_counts:
                            all_word_counts[word] = 0
                        all_word_counts[word] += 1
    print("finished counting")

    V = min(V, len(all_word_counts))
    all_word_counts = sorted(all_word_counts.items(), key=lambda x: x[1], reverse=True)

    top_words = [w for w, count in all_word_counts[:V-1]] + ['<UNK>']
    word2index = {w:i for i, w in enumerate(top_words)}
    unk = word2index['<UNK>']

    sents = []
    for f in files:
        for line in open(f):
            if line and line[0] not in '[*-|=\{\}':
                s = remove_punctuation(line).lower().split()
                if len(s) > 1:
                    # if a word is not nearby another word, there won't be any context!
                    # and hence nothing to train!
                    sent = [word2index[w] if w in word2index else unk for w in s]
                    sents.append(sent)
    return sents, word2index

# get the data
def train_model(savedDir):
    sentences, word2index = get_wiki() # can be replaced with get_brown()

    # number of unique words
    vocab_size = len(word2index)

    # config
    window_size = 5
    learnning_rate = 0.025
    final_learnning_rate = 0.0001
    num_negatives = 5 # number of negative samples to draw per input word
    epochs = 20
    D = 50 # word embedding size

    # learnning rate decay
    learnning_rate_delta = (learnning_rate - final_learnning_rate) / epochs

    # params
    W = np.random.randn(vocab_size, D) # input-to-hidden
    V = np.random.randn(D, vocab_size) # hidden-to-output

    # distribution for drawing negative samples
    p_neg = get_negative_sampling_distribution(sentences, vocab_size)

    # save the costs plot them per iteration
    costs = []

    # number of total words in corpus
    total_words = sum(len(sentences) for sentence in sentences)
    print("total number of words in corpus:", total_words)

    # for subsampling each sentence
    threshold = 1e-5
    p_drop = 1 - np.sqrt(threshold / p_neg)

    # train the model
    for epoch in range(epochs):
        # randomly order sentences so we don't always see
        # sentences in  the same order
        np.random.shuffle(sentences)

        # accumulate the cost
        cost = 0
        counter = 0
        t0 = datetime.now()
        for sentence in sentences:
            # keep only certain words based on p_neg
            sentence = [w for w in sentence \
                if np.random.random() < (1 - p_drop[w])
            ]
            if len(sentence) < 2:
                continue

            # randomly order words so we don't always see
            # samples in the same order
            randomly_ordered_positions = np.random.choice(
                len(sentence),
                size=len(sentence), # np.random.randint(1, len(sentence)+1)
                replace=False
            )

            for pos in randomly_ordered_positions:
                # the middle word
                word = sentence[pos]

                # get the positive context words/negative samples
                context_words = get_context(pos, sentence, window_size)
                neg_word = np.random.choice(vocab_size, p=p_neg)
                targets = np.array(context_words)

                # do one iteration of stochastic gradient descent
                c = sgd(word, targets, 1, learnning_rate, W, V)
                cost += c
                c = sgd(neg_word, targets, 0, learnning_rate, W, V)
                cost += c

            counter += 1
            if counter % 100 == 0:
                sys.stdout.write("processed %s / %s\r" % (counter, len(sentences)))
                sys.stdout.flush()
                # break

        # print stuff so we don't stare at a blank screen
        dt = datetime.now() - t0
        print("epoch complete:", epoch, "cost:", cost, "dt:", dt)

        # save the cost
        costs.append(cost)

        # update the learnning rate
        learnning_rate -= learnning_rate_delta

    # plot the cost per iteration
    plt.plot(costs)
    plt.show()

    # save model
    if not os.path.exists(savedDir):
        os.mkdir(savedDir)

    with open('%s/word2index.json' % savedDir, 'w') as f:
        json.dump(word2index, f)

    np.savez('%s/weights.npz' % savedDir, W, V)

    # return the model
    return word2index, W, V

# Todo(rest of functions must be implemented such as sdg(stochastic gradient descent)