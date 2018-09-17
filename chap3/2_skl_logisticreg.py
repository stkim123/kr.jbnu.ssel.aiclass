
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from mylib.plotdregion import plot_decision_region

if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=0)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # ml = Perceptron(eta0=0.01, max_iter=40, tol=0,  random_state=0)
    ml = LogisticRegression(C=1000.0, random_state=0)
    ml.fit(X_train_std, y_train)
    y_pred = ml.predict(X_test_std)
    print('총 테스트 개수: %d, 오류개수:%d' % (len(y_test), (y_test != y_pred).sum()))
    print('정확도: %.2f'%accuracy_score(y_test, y_pred))

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_region(X=X_combined_std, y=y_combined, classifier=ml,
                         test_idx=range(105, 150), title='scikit-learn Logistic Regression')