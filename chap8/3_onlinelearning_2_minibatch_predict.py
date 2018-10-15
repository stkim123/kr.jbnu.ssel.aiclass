

import numpy as np
import pickle
import os
from chap8.mylib.sgd_vectorizor import vect

curDir = os.getcwd()
clf = pickle.load(open (os.path.join(curDir, 'classifier', 'SGD_classifier.pkl'), 'rb'))

label={0:'부정적 의견', 1:'긍정적 의견'}
# example=['I love this movie.']
example=['This movie is really best ever in my entire life.']
# example=['This hate this movie']

X = vect.transform(example)
print('예측: %s\n 확률: %.3f%%' %
      (label[clf.predict(X)[0]], np.max(clf.predict_proba(X))*100))

