import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mylib.perceptron import Perceptron

if __name__ == '__main__' :
    df = pd.read_csv('iris.data', header=None)
    y = df.iloc[0:100, 4].values
    y = np.where(y=='Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values

    plt.scatter(X[:50, 0], X[:50, 1], color='r', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='b', marker='x', label='versicolor')
    plt.xlabel('RO! 210(cm)')
    plt.ylabel(' 20(cm)')
    plt.legend(loc=4)
    plt.show()

    ppn1 = Perceptron(eta=0.1)
    ppn1.fit(X, y)
    print(ppn1.errors_)
