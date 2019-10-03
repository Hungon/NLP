<<<<<<< HEAD
<h3>Negative Sampling</h3>
<p>Most of time, a word is not the target.<br>
Just take a sample of the "wrong words".</p>
<img src="https://user-images.githubusercontent.com/17066776/66144110-32bdaf00-e643-11e9-83c8-f6927b58c282.png" width="200" height="400" />
<p>Instead of doging multiclass cross-entropy,
just do binary cross-entropy on the negative samples.</p>
<img src="https://user-images.githubusercontent.com/17066776/66144287-86c89380-e643-11e9-9014-319410d95624.png" width="200" height="50" />
=======
<h3>Hierarchical Softmax</h3>
<p>1) At every node, we decide whether to go left or right using sigmoid.</p>
<img src="https://user-images.githubusercontent.com/17066776/66058358-371b9680-e575-11e9-8116-b55fbfec4750.png" width="150" height="30"/>
<img src="https://user-images.githubusercontent.com/17066776/66057437-a7291d00-e573-11e9-8182-09ceff05340f.png" width="200" height="200" />
<p>2) Output probability is product of every probability on the path to that word.</p>
<img src="https://user-images.githubusercontent.com/17066776/66059094-6c74b400-e576-11e9-8ee0-16ff0e731222.png" width="220" height="40"/>
<img src="https://user-images.githubusercontent.com/17066776/66059075-62eb4c00-e576-11e9-857c-4fb993873950.png" width="200" height="200" />
<p>Ex:<br>p(cat) = 0.4<br>p(racoon) = 0.6 * 0.7 = 0.42<br>p(dog) = 0.6 * 0.3 * 0.1 = 0.018<br>p(apple) = 0.6 * 0.3 * 0.9 = 0.162<br>Sum = 1, just as softmax should.</p>
<p>3) Since we have a binary tree, there's no need for a D * V ouutput matrix.</p>
<img src="https://user-images.githubusercontent.com/17066776/66060631-f02fa000-e578-11e9-887e-1e64857f9a00.png" width="220" height="40"/>
<p>
<li>The fact that input weight is V * D is not a problem, since we only index it.</li>
<li>For the output matrix in a naive bigram, we must multiply, because j and W are arbitrary dense matrices.</li>
<li>Now, each split has 1 vector of length D.</li>
<li>We still have O(V) output weights, but only O(logV) operations.</li>
>>>>>>> cf003d42e9fd3a465638e82bf2d86c9496eb7398
