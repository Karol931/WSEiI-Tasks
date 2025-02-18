import numpy as np
import matplotlib.pyplot as plt
from task1 import mlp, tanh, sigmoid
import math


def sigmoid_diff(y, beta):
    return beta*y*(1-y)

def tanh_diff(y, beta):
    return beta*(1-y*y)

def train_sample(x, d, eta, beta, w1, w2):
    y, v = mlp(x, w1, w2, beta)
    w2_grad = (y-d)
    w2 -= eta * w2_grad

    v_t = [v[i] for i in range(1,len(v))]
    w2_t = [w2[i] for i in range(1,len(w2))]

    g = tanh_diff(tanh(np.dot(w2_t, v_t),beta), beta)
    f = sigmoid_diff(sigmoid(np.dot(w1, x),beta), beta)
    w1_grad = np.dot(np.dot(np.dot(np.dot((y-d),g),w2_t),f),x)
    w1 -= eta*w1_grad

    return w1, w2, w1_grad, w2_grad

def train_epoch(xx, dd, eta, beta, w1, w2):
    sum_w1_grad = 0
    sum_w2_grad = 0
    for x, d in zip(xx, dd):
        w1, w2, w1_grad, w2_grad = train_sample(x, d,eta,beta, w1, w2)
        # print(w1, w2)
        sum_w1_grad += w1_grad
        sum_w2_grad += w2_grad
    
    return sum_w1_grad, sum_w2_grad

def check_errors(xx, w1, w2, beta, dd):
    err = 0
    for x, d in zip(xx,dd):
        y, _ = mlp(x, w1, w2, beta)
        if y < 0.1:
            y = 0
        elif y > 0.9:
            y = 1

        if y != d:
            err += abs(y-d)

    return err

def main():
    xx = np.array([[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]])
    dd = np.array([0,1,1,0])
    eta = 0.01
    w1 = np.random.randn(2,3)
    w2 = np.random.randn(3)
    beta = 1
    
    counter = 0
    errors = []
    while True:

        sum_w1_grad, sum_w2_grad = train_epoch(xx, dd, eta, beta, w1, w2)
        
        err = check_errors(xx, w1, w2, beta, dd)
        errors.append(err)
        
        if err == 0 or counter == 100000:
            break

        w1 -= eta*sum_w1_grad
        w2 -= eta*sum_w2_grad  

        counter += 1

    
    print(errors)
    plt.plot(errors)
    plt.ylabel('error')
    plt.xlabel('epoch')
    plt.show()

main()