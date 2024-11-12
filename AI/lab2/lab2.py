from perceptron import Perceptron
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from PIL import Image

def get_data(path):
    df = pd.read_csv(path, delimiter=',', header=None)
    y_data = df.iloc[:, :1]
    x_data = df.iloc[:, 1:]
    # print(df, y_data, x_data)

    return x_data, y_data


def choose_numbers(num1, num2, x_data, y_data):
    indexes_to_delete = y_data[~y_data[0].isin([num1,num2])].index

    y_data = y_data.drop(indexes_to_delete)
    y_data = y_data.reset_index(drop=True)

    x_data = x_data.drop(indexes_to_delete)
    x_data = x_data.reset_index(drop=True)

    y_data = y_data[0].replace({num1: 1, num2: 0})
    return x_data, y_data


def draw_number(number_pixels):
    n = int(np.sqrt(len(number_pixels)))
    number_pixels = number_pixels.reshape(n,n)
    plt.imshow(number_pixels)
    plt.show()


def normalize_data(x_data):
    x_data = (1.0*x_data)/x_data.values.max()
    
    return x_data


def main():
    TEST_DATA_PATH = 'mnist_test.csv'
    TRAIN_DATA_PATH = 'mnist_train.csv' 

    1
    x_train , y_train = get_data(TRAIN_DATA_PATH)
    x_test, y_test = get_data(TEST_DATA_PATH)

    #2
    num1, num2 = 3, 4
    x_train , y_train = choose_numbers(num1, num2, x_train, y_train)
    x_test, y_test = choose_numbers(num1, num2, x_test, y_test)

    #3
    for _ in range(5):
        rand = np.random.randint(0, len(x_train))
        draw_number(x_train.iloc[rand, :].to_numpy())

    #4
    x_train = normalize_data(x_train)
    x_test = normalize_data(x_test)

    #5
    perceptron = Perceptron(n=len(x_train.iloc[0,:]))

    #6
    errors = perceptron.train(xx=x_train.to_numpy(), d=y_train.to_numpy(), eta=0.01, tol=0)
    plt.plot(errors)
    plt.show()

    #7
    error_number, _ = perceptron.evaluate_test(x_test.to_numpy(), y_test.to_numpy())
    print(f'Liczba pomyłek dla zbioru testowego: {error_number}')


    #8
    y_predicted = [perceptron.predict(x) for x in x_test.to_numpy()]

    conf_matrix = confusion_matrix(y_test.to_numpy(), y_predicted)
    print(f'Macierz pomyłek:\n {conf_matrix}')

    #9
    print(f'accuracy: {accuracy_score(y_test.to_numpy(), y_predicted)}')
    print(f'precission: {precision_score(y_test.to_numpy(), y_predicted)}')
    print(f'recall: {recall_score(y_test.to_numpy(), y_predicted)}')
    print(f'f1_score: {f1_score(y_test.to_numpy(), y_predicted)}')

    #10
    # It should be 4
    im = Image.open('./number.png').convert('L')
    im = im.resize((28,28))
    x = list(im.getdata())
    
    x = (1.0*np.asarray(x))/max(x)
    y_pred = perceptron.predict(x)
    print(f'Liczba z obrazka to: {4 if y_pred == 0 else 3}')

if __name__ == '__main__':
    main()