import numpy as np

class Perceptron:
    
    def __init__(self, n, bias=True):
        self.n = n
        self.w = np.random.randn(n)
        self.b = 1.0 if bias else 0


    def predict(self, x):
        y = 1 if (np.dot(self.w,x) + self.b) > 0 else 0
        
        return y


    def train(self, xx, d, eta, tol):
        while True:
            for x, di in zip(xx, d):
                y = self.predict(x)
                if y == 1 and di == 0:
                    self.w -= np.multiply(eta,x)
                    self.b -= eta
                elif y == 0 and di == 1:
                    self.w += np.multiply(eta,x)
                    self.b += eta

            error, _ = self.evaluate_test(xx, d)
            print(error)
            if tol >= error:
                break


    def evaluate_test(self, xx, d):
        y = [self.predict(x) for x in xx]
        err = np.sum(abs(di-yi) for di, yi in zip(d, y))

        return err, y