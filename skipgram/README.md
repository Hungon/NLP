<h3>Hierarchical Softmax</h3>
<p>1) At every node, we decide whether to go left or right using sigmoid.</p>
<img src="https://user-images.githubusercontent.com/17066776/66058358-371b9680-e575-11e9-8116-b55fbfec4750.png" width="150" height="30"/>
<img src="https://user-images.githubusercontent.com/17066776/66057437-a7291d00-e573-11e9-8182-09ceff05340f.png" />
<p>2) Output probability is product of every probability on the path to that word.</p>
<img src="https://user-images.githubusercontent.com/17066776/66059094-6c74b400-e576-11e9-8ee0-16ff0e731222.png" width="220" height="40"/>
<img src="https://user-images.githubusercontent.com/17066776/66059075-62eb4c00-e576-11e9-857c-4fb993873950.png" />
<p>Ex:<br>p(cat) = 0.4<br>p(racoon) = 0.6 * 0.7 = 0.42<br>p(dog) = 0.6 * 0.3 * 0.1 = 0.018<br>p(apple) = 0.6 * 0.3 * 0.9 = 0.162<br>Sum = 1, just as softmax should.</p>
