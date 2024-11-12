import numpy as np

class Perceptron:
    # Inicjalizator, ustawiający atrybut self.w oraz self.b jako wektor losowych wag, n ilość sygnałów wejściowych
    def __init__(self, n, bias=True):
        self.n = n
        self.w = np.random.randn(n)
        self.b = 1.0 if bias else 0

    # Metoda obliczająca odpowiedz modelu dla zadanego sygnału wejściowego x=[x1,x2,...,xN]
    def predict(self, x):
        y = 1 if (np.dot(self.w,x) + self.b) > 0 else 0
        
        return y

    # Metoda uczenia według reguły perceptronu, xx - zbiór danych uczących, d - odpowiedzi,
    # eta - współczynnik uczenia,
    # tol - tolerancja (czyli jak duży błąd jesteśmy w stanie zaakceptować)
    def train(self, xx, d, eta, tol):
        err = []
        while True:
            for x, di in zip(xx, d):
                # print(x, d)
                y = self.predict(x)
                if y == 1 and di == 0:
                    self.w -= np.multiply(eta,x)
                    self.b -= eta
                elif y == 0 and di == 1:
                    self.w += np.multiply(eta,x)
                    self.b += eta

            error, _ = self.evaluate_test(xx, d)
            
            err.append(error)
            if tol >= error:
                break

        return err

    # Metoda obliczająca błąd dla danych testowych xx
    # zwraca błąd oraz wektor odpowiedzi perceptronu dla danych testowych
    def evaluate_test(self, xx, d):
        y = [self.predict(x) for x in xx]
        err = np.sum(abs(di-yi) for di, yi in zip(d, y))

        return err, y