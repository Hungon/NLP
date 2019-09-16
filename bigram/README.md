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
<p>Simplicity. Many probability distibutions habe asn exponetial form. Taking the log these distributions eliminates the exponential function, unwrapping the exponent.