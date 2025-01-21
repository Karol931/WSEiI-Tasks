import numpy as np

def sigmoid(x, beta):
    return 1.0/(1.0+np.exp(-beta*x))

def tanh(x, beta):
    return np.tanh(beta*x)

def mlp(x, w1, w2, beta):
    v = sigmoid(np.dot(w1, x), beta)
    v = np.insert(v, 0, 1.0)
    y = tanh(np.dot(w2, v), beta)
    return y, v