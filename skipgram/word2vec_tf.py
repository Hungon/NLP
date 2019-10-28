from __future__ import print_function, division
from builtins import range


import json
# System install
# pip3 install --user --upgrade tensorflow  # install in $HOME
# Virtual install
# pip install --upgrade tensorflow==1.15.0
# Verify the install
# python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
import tensorflow as tf
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
    samples_per_epoch = int(1e5)
    epochs = 10
    D = 50 # word embedding size

    # learnning rate decay
    learnning_rate_delta = (learnning_rate - final_learnning_rate) / epochs

    # distribution for drawing negative samples
    p_neg = get_negative_sampling_distribution(sentences, vocab_size)

    # params
    W = np.random.randn(vocab_size, D).astype(np.float32) # input to hidden
    V = np.random.randn(D, vocab_size).astype(np.float32) # hidden to output

    # create the model
    tf_input = tf.placeholder(tf.int32, shape=(None,))
    tf_negword = tf.placeholder(tf.int32, shape=(None,))
    tf_context = tf.placeholder(tf.int32, shape=(None,)) # targets (context)
    tfw = tf.Variable(W)
    # T as transpose
    tfv = tf.Variable(V.T)
    # biases = tf.Variable(np.zeros(vocab_size, dtype=np.float32))

    def dot(A, B):
        C = A * B
        return tf.reduce_sum(C, axis=1)

    # correct middle word output
    emb_input = tf.nn.embedding_lookup(tfw, tf_input) # 1 * D
    emb_output = tf.nn.embedding_lookup(tfv, tf_context) # N * D
    correct_output = dot(emb_input, emb_output) # N
    pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones(tf.shape(correct_output)), logits=correct_output)

    # inicorrect middle word output
    emb_input = tf.nn.embedding_lookup(tfw, tf_negword)
    incorrect_output = dot(emb_output, emb_output)
    neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros(tf.shape(incorrect_output)), logits=incorrect_output)

    # total loss
    loss = tf.reduce_mean(pos_loss) + tf.reduce_mean(neg_loss)

    # output = hidden.dot(tfV)

    # loss
    # neither of the built-in TF functions work well
    # per_sample_loss = tf.nn.nce_loss(
    # # per_sample_loss = tf.nn.sampled_softmax_loss(
    #   weights=tfV,
    #   biases=biases,
    #   labels=tfY,
    #   inputs=hidden,
    #   num_sampled=num_negatives,
    #   num_classes=vocab_size,
    # )
    # loss = tf.reduce_mean(per_sample_loss)

    # optimizer
    # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    train_op = tf.train.MomentumOptimizer(0.1, momentum=0.9).minimize(loss)

    # make sessiion
    session = tf.Session()
    init_op = tf.global_variables_initializer()
    session.run(init_op)

    # save the costs to plot them per iteration
    costs = []

    # number of total words in corpus
    total_words = sum(len(sentence) for sentence in sentences)
    print("total number of words in corpus:", total_words)

    # for subsampling each sentence
    threshold = 1e-5
    p_drop = 1 - np.sqrt(threshold / p_neg)

    t0 = datetime.now()
    # train the model
    for epoch in range(epochs):
        # randomly order sentences so we do not always see
        # sentences in the same order
        np.random.shuffle(sentences)

        # acumulate the cost
        cost = 0
        counter = 0
        inputs = []
        targets = []
        negwords = []
        t1 = datetime.now()
        for sentence in sentences:
            # keep only certain words baed on p_neg
            sentence = [w for w in sentence \
                if np.random.random() < (1 - p_drop[w])
            ]
            if len(sentence) < 2:
                continue
            
            randomly_ordered_positions = np.random.choice(len(sentence),size=len(sentence), replace=False)

            for j, pos in enumerate(randomly_ordered_positions):
                # the middle word
                word = sentence[pos]
                # get the positive contexr words/negative samples
                context_words = get_context(pos, sentence, window_size)
                neg_word = np.random.choice(vocab_size, p=p_neg)

                n = len(context_words)
                inputs += [word]*n
                negwords += [neg_word]*n
                targets += context_words

                if len(inputs) >= 128:
                    _, c = session.run((train_op, loss),feed_dict={tf_input: inputs, tf_negword: negwords, tf_context: targets})
                    cost += c
                    # reset
                    inputs = []
                    targets = []
                    negwords = []
                counter += 1
                if counter % 100 == 0:
                    # sys.stdout.write("processed %s / %s\n" % (counter, len(sentences)))
                    sys.stdout.flush()
                    # break
                
        # prinit stuff so we do not stare at a blank screen
        dt = datetime.now() - t1
        print("epoch complete: %s / %s" % (epoch+1, epochs), "cost:", cost, "dt:", dt)
        # save the cost
        costs.append(cost)
        # update the learning rate
        learnning_rate_delta -= learnning_rate_delta

    print("completed: %s / %s" % (epoch+1, epochs), "cost:", cost, "total time:", (datetime.now()-t0))
    # plot the cost per iteration
    plt.plot(costs)
    plt.show()
    # get the params
    W, VT = session.run((tfw, tfv))
    V = VT.T

    # save the model
    if not os.path.exists(savedDir):
        os.mkdir(savedDir)
    with open('%s/word2index.json'%savedDir, 'w') as f:
        json.dump(word2index, f)
    np.savez('%s/weights.npz' % savedDir, W, V)

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
  word2index, W, V = train_model('w2v_tf')
  # word2index, W, V = load_model('w2v_tf')
  test_model(word2index, W, V)
