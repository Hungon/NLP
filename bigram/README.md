<h4>A model with only bigrams</h4>
<li>Perhaps the sentence: "The quick brown fox jumps over the lazy turtle" never apperas in our corpus.</li>
<li>Nor word:</li>
<p>"The quick brown fox jumps over the lazy cat"</p>
<p>"The quick brown fox jumps over the lazy rabbit"</p>
<li>But phrases like "lazy turtle", "lazy cat" and "lazy rabbit" probobaly do</li>
<li>It's easier to model the probabiliity of shorter phrases because we have more samples, and this makes a realistiic sentence like the onse above more probable.</li>
<h4>Point</h4>
<li>Make use of log probability</li>
<p>1. Speed. Since multiplication is more expensive than addition, taking the product of a high number of probablities is often faster if they are represented in log form.</p>
<p>2. Accuracy. The use of log probablities improves numerical stability, when the probablities are very small, because of the way in which computes approximate real numbers.</p>
<p>Simplicity. Many probability distibutions have an exponential form. Taking the log these distributions eliminates the exponential function, unwrapping the exponent.

<h3>Bigram probabilities</h3>
<li>Since x just contains 0s and 1s we can ignore</li>
<li>Matrix multiplication rule:(V*D)(D*V) -> (V*V)</li>
<code>p(y | x) = softmax(W&supT;x)</code><span>as described in logistic.py.</span>
<code>p(y | x) = softmax(W2&supT; tanh(W1&supT;x))</code><span>as described in neural_network.py</span>
