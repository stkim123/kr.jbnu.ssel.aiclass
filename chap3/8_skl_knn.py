

from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

from mylib.plotdregion import plot_decision_region

if __name__ == '__main__':

    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=0)

    plt.scatter(X[y==1, 0], X[y==1, 1], color = 'b', marker='x', label='1')
    plt.scatter(X[y==-1, 0], X[y==-1, 1], color = 'r', marker='s', label='-1')
    plt.ylim(-3.0)
    plt.legend()
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=0)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # ml = Perceptron(eta0=0.01, max_iter=40, tol=0,  random_state=0)
    # ml = LogisticRegression(C=1000.0, random_state=0)
    # ml = SVC(kernel='linear', C=1.0, random_state=0)
    # ml = SGDClassifier(loss = 'perceptron')
    # ml = SGDClassifier(loss='log')
    # ml = SGDClassifier(loss='hinge')
    # ml = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
    # ml = RandomForestClassifier(criterion='entropy', n_estimators=10, n_jobs=2, random_state=1)

    ml = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')

    ml.fit(X_train_std, y_train)
    y_pred = ml.predict(X_test_std)
    print('총 테스트 개수: %d, 오류개수:%d' % (len(y_test), (y_test != y_pred).sum()))
    print('정확도: %.2f' % accuracy_score(y_test, y_pred))

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_region(X=X_combined_std, y=y_combined, classifier=ml,
                         test_idx=range(105, 150), title='Random Forest Classifier')

