from task1 import Perceptron
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def organise_2d_data(path):
    df = pd.read_csv(path, skiprows=1, delimiter=';', names=['X1', 'X2', 'L'], decimal=',')
    df.head()
    train, test = train_test_split(df, test_size=0.2)
    x_train = train[['X1', 'X2']].values.tolist()
    y_train = train['L'].values.tolist()
    x_test = test[['X1', 'X2']].values.tolist()
    y_test = test['L'].values.tolist()

    return x_train, y_train, x_test, y_test


def organise_3d_data(path):
    df = pd.read_csv(path, skiprows=1, delimiter=';', names=['X1', 'X2', 'X3', 'L'], decimal=',')
    df.head()
    train, test = train_test_split(df, test_size=0.2)
    x_train = train[['X1', 'X2', 'X3']].values.tolist()
    y_train = train['L'].values.tolist()
    x_test = test[['X1', 'X2', 'X3']].values.tolist()
    y_test = test['L'].values.tolist()

    return x_train, y_train, x_test, y_test


def draw_data_2d(y, x, perceprton):
    x_axis = [xi[0] for xi in x]
    y_axis = [xi[1] for xi in x]
    plt.scatter(x=x_axis, y=y_axis, c=y)
    x_values = np.linspace(np.min(x_axis), np.max(x_axis), 100)
    y_values = -(perceprton.w[0] * x_values + perceprton.b) / perceprton.w[1]
    plt.plot(x_values, y_values)
    plt.show()

def draw_data_3d(y, x, perceprton):
    x_axis = [xi[0] for xi in x]
    y_axis = [xi[1] for xi in x]
    z_axis = [xi[2] for xi in x]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x_axis,y_axis,z_axis,c=y)
    x_values = np.linspace(np.min(x_axis), np.max(x_axis),10)
    y_values = np.linspace(np.min(y_axis), np.max(y_axis),10)
    x, y = np.meshgrid(x, y)
    z = -(perceprton.w[0] * x + perceprton.w[1] * y + perceprton.b) / perceprton.w[2]
    ax.plot_surface(x,y,z)
    plt.show()

def data_2d_classification(eta, tol, n, path):
    x_train, y_train, x_test, y_test = organise_2d_data(path)

    perceprton = Perceptron(n=n)
    perceprton.train(x_train, y_train, eta=eta, tol=tol)

    errors, y = perceprton.evaluate_test(x_test, y_test)
    print(f'Errors: {errors}')

    return y, x_test, perceprton


def data_3d_classification(eta, tol, n, path):
    x_train, y_train, x_test, y_test = organise_3d_data(path)

    perceprton = Perceptron(n=n)
    perceprton.train(x_train, y_train, eta=eta, tol=tol)
    
    errors, y = perceprton.evaluate_test(x_test, y_test)
    print(f'Errors: {errors}')

    return y, x_test, perceprton


def main():
    ETA = 0.1
    TOL = 0
    N_2D = 2
    N_3D = 3
    PATH_2D = './2D.csv'
    PATH_3D = './3D.csv'

    y_2d, x_2d, perceptron_2d = data_2d_classification(eta=ETA, tol=TOL, n=N_2D, path=PATH_2D)
    draw_data_2d(y_2d, x_2d, perceptron_2d)

    y_3d, x_3d, perceptron_3d = data_3d_classification(eta=ETA, tol=TOL, n=N_3D, path=PATH_3D)
    draw_data_3d(y_3d, x_3d, perceptron_3d)

if __name__ == '__main__':
    main()