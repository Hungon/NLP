from __future__ import print_function, division
import os
import numpy as np
from builtins import range, input
from future.utils import iteritems
import sys
sys.path.append(os.path.abspath('..'))

from rnn.corpus import get_sentences_with_word2idx_limit_vocab, get_sentences_with_word2idx

# Note: you may need to update your version of future
# sudo pip install -U future


def get_bigram_probs(sentences, V, start_index, end_index, smoothing=1):
    # structure of bigram probablity matriix will be:
    # (last word, current word) -> probablity
    # we will use add-1 smoothing
    # note: we'll always ignore this from the End token
    bigram_probs = np.ones((V, V)) * smoothing
    for sentence in sentences:
        for i in range(len(sentence)):
            if i == 0:
                # beginning word
                bigram_probs[start_index, [sentence[i]]] += 1
            else:
                # middle word
                bigram_probs[sentence[i-1], sentence[i]] += 1

            # if we're at the final word we update the bigram for last -> current
            # and current -> end token
            if i == len(sentence) - 1:
                # final word
                bigram_probs[sentence[i], end_index] += 1
    # normalize the counts along the rows to get probalitiies
    bigram_probs /= bigram_probs.sum(axis=1, keepdims=True)
    return bigram_probs


if __name__ == '__main__':
    # load in the data
    # note: sentences are already converted to sequences of word indexes
    # note: you can limit the vacab size if you run out of memory
    sentences, word2index = get_sentences_with_word2idx_limit_vocab(
        10000)
    # sentences, word2index = get_sentences_with_word2idx()

    # vocab size
    V = len(word2index)
    print("Vacab size:", V)
    start_index = word2index['START']
    end_index = word2index['END']

    # we will also treat beginning of sentence and end of sentence as biigram
    # Start -> first word
    # last word -> End

    # a matrix where:
    # row = last word
    # col = current word
    # value at [row, col] = p(current word | last word)
    bigram_probs = get_bigram_probs(
        sentences, V, start_index, end_index, smoothing=0.1)

    # a function calculate normalized log prob score for a sentence
    def get_score(sentence):
        score = 0
        for i in range(len(sentence)):
            if i == 0:
                # beginning word
                score += np.log(bigram_probs[start_index, sentence[i]])
            else:
                # middle word
                score += np.log(bigram_probs[sentence[i-1], sentence[i]])

        # final word
        score += np.log(bigram_probs[sentence[-1], end_index])

        # normalize the score
        return score / (len(sentence) + 1)

    # a function to map word indexes back to real word
    index2word = dict((v, k) for k, v in iteritems(word2index))

    def get_words(sentence):
        return ' '.join(index2word[i] for i in sentence)

    # when we sample a fake sentence, we want to ensure not to sample
    # start token or end token
    sample_probs = np.ones(V)
    sample_probs[start_index] = 0
    sample_probs[end_index] = 0
    sample_probs /= sample_probs.sum()

    # test our model on real and fake sentences
    while True:
        # real sentence
        real_index = np.random.choice(len(sentences))
        real = sentences[real_index]

        # fake sentence
        fake = np.random.choice(V, size=len(real), p=sample_probs)

        print("Real:", get_words(real), "Score:", get_score(real))
        print("Fake:", get_words(fake), "Score:", get_score(fake))

        # input you own sentence
        custom = input("Enter your own sentence\n")
        custom = custom.lower().split()

        # check that all tokens exist in word2index otherwise, we can't get score
        bad_sentence = False
        for token in custom:
            if token not in word2index:
                bad_sentence = True

        if bad_sentence:
            print("Sorry, you entered words that are not in the vacabulary")
        else:
            # convert sentent into list of indexes
            custom = [word2index[token] for token in custom]
            print("Score:", get_score(custom))

        cont = input("Continue? [Y/N]")
        if cont and cont.lower() in ('N', 'n'):
            break
