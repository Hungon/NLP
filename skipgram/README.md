<h3>Negative Sampling</h3>
<p>Most of time, a word is not the target.<br>
Just take a sample of the "wrong words".</p>
<img src="https://user-images.githubusercontent.com/17066776/66144110-32bdaf00-e643-11e9-83c8-f6927b58c282.png" width="200" height="400" />
<p>Instead of doging multiclass cross-entropy,
just do binary cross-entropy on the negative samples.</p>
<img src="https://user-images.githubusercontent.com/17066776/66144287-86c89380-e643-11e9-9014-319410d95624.png" width="200" height="50" />
