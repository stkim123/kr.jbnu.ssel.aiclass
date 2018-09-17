import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mylib.adalinesgd import AdalineSGD

if __name__ == '__main__' :
    df = pd.read_csv('iris.data', header=None)
    y = df.iloc[0:100, 4].values
    y = np.where(y=='Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values

    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    adal = AdalineSGD(eta=0.01, n_iter=15, random_state=1).fit(X_std, y)
    plt.plot(range(1, len(adal.cost_) + 1), adal.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Average Cost')
    plt.title('Adaline Stochastic GD – Learning rate 0.01')
    plt.show()





