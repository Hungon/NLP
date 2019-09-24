# NLP
I'm gonna look at word2vec, Glove, word embedings, and sentiment analysis with recursive nets.

<h3>Pretrained data</h3>

<li>Sentiment Analysis</li>
<p>you can download sentiment data set from <a href="https://nlp.stanford.edu/sentiment/">stanford</a>.</p>

<li>Word Vector</li>
<p>you can download pretrained word vector from <a href="https://nlp.stanford.edu/projects/glove/">stanford</a>.</p>

<p>Dump file from <a href="https://dumps.wikimedia.org/enwiki/">wikipedia.</a></p>

<p>Dump file is converted into plain text using <a href="https://github.com/yohasebe/wp2txt">wp2tex.</a></p>

<h3>Model</h3>
<h4>Markov Assumption</h4>
<p>Markov assumption is that whatever you see now depends only on what you saw in the previous step.<br>
In other words any older terms just disapper.</p>
<code>p(Wn|Wn-1, Wn-2, ..., W1) = p(Wn | Wn-1)</code>
<p>Also See <a target="_blank" href="https://en.wikipedia.org/wiki/Markov_chain">Markov Chain</a></p>
<h4>Bigram</h4>
<code>p(Wn|Wn-1)</code>
<p>This is a sequance of two adjacent elements from a string of tokens, which are typically letters, syllables or words.</p>
<h4>Unigram</h4>
<p>Contiguous sequence of n items from a given sample of text or speech.</p>
<code>p(A = count(A) devided by corpus length</code>
<h4>Trigram</h4>
<p>Trigam is a special case of the n-gram, where n is 3.</p>
<code>p(C | A, B) = count(A->B->C) devided by count(A->B)</code>

<h4>Reference</h4>

<p>About <a href="http://www.tfidf.com/">TF-IDF</a>(stands for term frequency-inverse document frequency)</p>

<p>About <a href="https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html">TSNE</a></p>

<p>About <a href="https://en.wikipedia.org/wiki/Word2vec#CBOW_and_skip_grams">CBOW</a>(stands for continuous bag-of-words)</p>

<p>About <a href="https://towardsdatascience.com/skip-gram-nlp-context-words-prediction-algorithm-5bbf34f84e0c">Skip-gram</a>(opposite of CBOW)</p>
