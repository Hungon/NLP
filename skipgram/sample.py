from __future__ import print_function, division
from builtins import range
import numpy as np

N = 100
D = 2

X = np.random.randn(N, D)

# center the first 50 points at (-2, -2)
X[:50,:] = X[:50,:] - 2 * np.ones((50,D))
# center the last 50 points at (2, 2)
X[:50,:] = X[:50,:] + 2 * np.ones((50,D))
# labels: first 50 are 0, last 50 are 1
T = np.array([0]*50 + [1]*50)

# add a colum of ones
# ones = np.array([[1]*N]).T
ones = np.ones((N, 1))
Xb = np.concatenate((ones, X), axis=1)

# randomly initialize the weights
w = np.random.randn(D + 1)

# calculate the model output
z = Xb.dot(w)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

Y = sigmoid(z)

# calculate the cross-entropy error
def crossz_entropy(T, Y):
    E = 0
    for i in range(len(T)):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E

# let's do gradient descen 100 times
learning_rate = 0.1
for i in range(100):
    if i % 10 == 0:
        print(crossz_entropy(T, Y))

    # gradient descen weight update with regularization
    w += learning_rate * (Xb.T.dot(T - Y) - 0.1 * w)

    # recalculate Y
    Y = sigmoid(Xb.dot(w))

print("Final w:", w)
