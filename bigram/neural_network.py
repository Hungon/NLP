from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime

import os
import sys
sys.path.append(os.path.abspath('..'))
# from rnn.util import get_wikipedia_data
from rnn.corpus import get_sentences_with_word2idx_limit_vocab, get_sentences_with_word2idx

from markov import get_bigram_probs

if __name__ == '__main__':
    
    limiited_vocab_size = 20
    # load in the data
    # note: sentences are alredy converted into sequences of word indexes
    # note: you can limit the vocab size if you out of memory
    sentences, word2index = get_sentences_with_word2idx_limit_vocab(limiited_vocab_size)
    # sentences, word2index = get_sentences_with_word2index()

    # vocab size
    V = len(word2index)
    print('Vocab size:', V)

    # we will also treat beginning of sentences and end of sentence as bigram
    # Start -> first word
    # last word -> End
    start_index = word2index['START']
    end_index = word2index['END']

    # a matrix where:
    # row = last word
    # col = current word
    # value at [row, col] = p(current word | last word)
    bigram_probs = get_bigram_probs(sentences, V, start_index, end_index, smoothing =0.1)

    # train a shallow neural network model
    D = 100
    W1 = np.random.randn(V, D) / np.sqrt(V)
    W2 = np.random.randn(D, V) / np.sqrt(D)

    losses = []
    epochs = 1
    lr = 1e-2

    def softmax(a):
        a = a - a.max()
        exp_a = np.exp(a)
        return exp_a / exp_a.sum(axis=1, keepdims=True)

    # what is the loss if we set W = log(bigram_probs)?
    W_bigram = np.log(bigram_probs)
    bigram_losses = []

    t0 = datetime.now()
    for epoch in range(epochs):
        # shuffle sentences at each epoch
        random.shuffle(sentences)

        j = 0 # keep track of iterations
        for sentence in sentences:
            # convert sentence into one-hot encoded inputs and targets
            sentence = [start_index] + sentence + [end_index]
            n = len(sentence)
            # do not one-hot encoded inputs and targets
            inputs = sentence[:n-1]
            targets = sentence[1:]
            
            # get output predictions
            hidden = np.tanh(W1[inputs])
            predictions = softmax(hidden.dot(W2))

            # keep track of the loss
            loss = -np.sum(np.log(predictions[np.arange(n - 1), targets])) / (n - 1)
            losses.append(loss)

            # do a gradient descent step
            # do it after loss since the calculation of doutput will overwrite predictions
            # we don't want to make a copy because it would be slow
            doutput = predictions # N * V
            doutput[np.arange(n - 1), targets] -= 1
            W2 = W2 - lr * hidden.T.dot(doutput) # (D * N)(N * V)
            dhidden = doutput.dot(W2.T) * (1 - hidden * hidden) # (N * V)(V * D)(N * D)
            
            # fastest way
            np.subtract.at(W1, inputs, lr * dhidden)

            # keep track of the bigram loss
            # only do it for the first epoch to avoid redundancy
            if epoch == 0:
                bigram_predictions = softmax(W_bigram[inputs])
                bigram_loss = -np.sum(np.log(bigram_predictions[np.arange(n - 1), targets])) / (n - 1)
                bigram_losses.append(bigram_loss)

            if j % 100 == 0:
                print("epoch", epoch, "sentence: %s/%s" % (j, len(sentences)), "loss", loss)
            j += 1

        print("Elapsed time trainning:", datetime.now() - t0)
        plt.plot(losses)

        # plt a horizontal line for the bigram loss
        avg_bigram_loss = np.mean(bigram_losses)
        print("ave_bigram_loss:", avg_bigram_loss)
        plt.axhline(y=avg_bigram_loss, color='r', linestyle='-')


        # plot smeehted losses to reduce variability
        def smoothed_loss(x, decay=0.99):
            y = np.zeros(len(x))
            last = 0
            for t in range(len(x)):
                z = decay * last + (1 - decay) * x[t]
                y[t] = z / (1 - decay ** (t + 1))
                last = z
            return y

        plt.plot(smoothed_loss(losses))
        plt.show()

        # plot W and bigram probs side-by-side
        # for the most common 200 words
        plt.subplot(1, 2, 1)
        plt.title("Neural Network Model")
        # compute hyperbolic tangent element-wise, 
        # multiply the result by W2 and show it
        plt.imshow(np.tanh(W1).dot(W2))
        plt.subplot(1, 2, 2)
        plt.title("Bigram Probs")
        plt.imshow(W_bigram)
        plt.show()

    
