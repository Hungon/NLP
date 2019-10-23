from __future__ import print_function, division
from builtins import range
import json
import numpy as np
import matplotlib.pyplot as plt
# python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
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
    sentences, word2index = get_brown() # can be replaced with get_brown()

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
    if p_neg.all() > 0:
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

def get_negative_sampling_distribution(sentences, vocab_size):
    # Pn(w) = prob of word occuring
    # we would like to sample the negative samples
    # such that words that occur more often
    # should be sampled more often
    word_freq = np.zeros(vocab_size)
    word_count = sum(len(sentence) for sentence in sentences)
    for sentence in sentences:
        for word in sentence:
            word_freq[word] += 1

    # smooth it
    p_neg = word_freq**0.75

    # normarize it
    p_neg = p_neg / p_neg.sum()

    # assert(np.all(p_neg > 0))
    return p_neg

def get_context(pos, sentence, window_size):
    # input:
    # a sentence of the form: xxxxx ccc pos ccc xxxx
    # output:
    # the context word indices: cccccc
    start = max(0, pos - window_size)
    end_ = min(len(sentence), pos + window_size)

    context = []
    for ctx_pos, ctx_word_index in enumerate(sentence[start:end_], start=start):
        if ctx_pos != pos:
            # do not include the input word itself as a target
            context.append(ctx_word_index)
    return context

def sgd(input_, targets, label, learning_rate, W, V):
    # W[input_] shape: D
    # V[:,targets] shape: D * N
    # activation shape: N
    # print("input_:", input_, "targets:", targets)
    activation = W[input_].dot(V[:,targets])
    prob = sigmoid(activation)

    # gradients
    gV = np.outer(W[input_], prob - label) # D * N
    gW = np.sum((prob - label) * V[:,targets], axis=1) # D

    V[:,targets] -= learning_rate * gV # D * N
    W[input_] -= learning_rate * gW # D

    # return cost(binary cross entropy)
    cost = label * np.log(prob = 1e-10) + (1 - label) * np.log(1 - prob + 1e-10)
    return cost.sum()

def load_model(savedDir):
    with open('%s/word2index.json' % savedDir) as f:
        word2index = json.load(f)
    npz = np.load('%s/weights.npz' % savedDir)
    W = npz['arr_0']
    V = npz['arr_1']
    return word2index, W, V

def analogy(pos1, neg1, pos2, neg2, word2index, index2word, W):
    V,D = W.shape

    # do not actually use pos2 in calculation just print what's excepted
    print("testing: %s - %s = %s - %s" % (pos1, neg1, pos2, neg2))
    for w in (pos1, neg1, pos2, neg2):
        if w not in word2index:
            print("Sorry, %s not in word2index" % w)
            return
    p1 = W[word2index[pos1]]
    n1 = W[word2index[neg1]]
    p2 = W[word2index[pos2]]
    n2 = W[word2index[neg2]]

    vec = p1 - n1 + n2

    distances = pairwise_distances(vec.reshape(1, D), W, metric='cosine').reshape(V)
    index = distances.argsort()[:10]

    # pick one that's not p1, n1 or n2
    best_index  = -1
    keep_out = [word2index[w] for w in (pos1, neg1, neg2)]
    print("keep out:", keep_out)
    for i in index:
        if i not in keep_out:
            best_index = i
            break
    print("best index:", best_index)

    print("got: %s - %s = %s - %s" % (pos1, neg1, index2word[best_index], neg2))
    print("closest 10:")
    for i in index:
        print(index2word[i], distances[i])

    print("dist to %s:" % pos2, cos_dist(p2, vec))
    print("\n")

def test_model(word2index, W, V):
    # there are multiple ways to get hte "final" word embedding
    index2word = {i:w for w, i in word2index.items()}
    # V.T is same as V.transpose()
    for We in (W, (W + V.T) / 2):
        print("**********")

        analogy('king', 'man', 'queen', 'woman', word2index, index2word, We)
        analogy('king', 'prince', 'queen', 'princess', word2index, index2word, We)
        analogy('miami', 'florida', 'dallas', 'texas', word2index, index2word, We)
        analogy('einstein', 'scientist', 'picasso', 'painter', word2index, index2word, We)
        analogy('japan', 'sushi', 'germany', 'bratwurst', word2index, index2word, We)
        analogy('man', 'woman', 'he', 'she', word2index, index2word, We)
        analogy('man', 'woman', 'uncle', 'aunt', word2index, index2word, We)
        analogy('man', 'woman', 'brother', 'sister', word2index, index2word, We)
        analogy('man', 'woman', 'husband', 'wife', word2index, index2word, We)
        analogy('man', 'woman', 'actor', 'actress', word2index, index2word, We)
        analogy('man', 'woman', 'father', 'mother', word2index, index2word, We)
        analogy('heir', 'heiress', 'prince', 'princess', word2index, index2word, We)
        analogy('nephew', 'niece', 'uncle', 'aunt', word2index, index2word, We)
        analogy('france', 'paris', 'japan', 'tokyo', word2index, index2word, We)
        analogy('france', 'paris', 'china', 'beijing', word2index, index2word, We)
        analogy('february', 'january', 'december', 'november', word2index, index2word, We)
        analogy('france', 'paris', 'germany', 'berlin', word2index, index2word, We)
        analogy('week', 'day', 'year', 'month', word2index, index2word, We)
        analogy('week', 'day', 'hour', 'minute', word2index, index2word, We)
        analogy('france', 'paris', 'italy', 'rome', word2index, index2word, We)
        analogy('paris', 'france', 'rome', 'italy', word2index, index2word, We)
        analogy('france', 'french', 'england', 'english', word2index, index2word, We)
        analogy('japan', 'japanese', 'china', 'chinese', word2index, index2word, We)
        analogy('china', 'chinese', 'america', 'american', word2index, index2word, We)
        analogy('japan', 'japanese', 'italy', 'italian', word2index, index2word, We)
        analogy('japan', 'japanese', 'australia', 'australian', word2index, index2word, We)
        analogy('walk', 'walking', 'swim', 'swimming', word2index, index2word, We)


if __name__ == '__main__':
  word2index, W, V = train_model('w2v_model')
  # word2index, W, V = load_model('w2v_model')
  test_model(word2index, W, V)
