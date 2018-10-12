

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import pickle
import os
from sent_analysis2.mylib.progbar import ProgBar
from sent_analysis2.mylib.sgd_tokenizer import sgd_tokenizer

def stream_docs(path):
    with open(path, 'r', encoding='UTF8') as f:
        next(f)
        for line in f:
            text, label = line[ : -3], int(line[-2])
            yield text, label

def get_minibatch(doc_stream, size) :
    docs, y = [], []
    try:
        for _ in range(size) :
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

vect = HashingVectorizer(decode_error = 'ignore', n_features = 2**21, tokenizer=sgd_tokenizer)

clf = SGDClassifier(loss='log', random_state=1, max_iter=1, tol=0)
doc_stream = stream_docs('./data/refined_movie_review.csv')

pbar = ProgBar(45)
classes = np.array([0,1])

for _ in range(45) :
    x_train, y_train = get_minibatch(doc_stream, size = 1000)
    if not x_train :
        break

    x_train = vect.transform(x_train)
    clf.partial_fit(x_train, y_train, classes = classes)
    pbar.update()

x_test, y_test = get_minibatch(doc_stream, size = 5000)
x_test = vect.transform(x_test)
print('정확도: %.3f' % clf.score(x_test, y_test))

curDir = os.getcwd()
dest = os.path.join(curDir, 'classifier')
if not os.path.exists(dest):
    os.makedirs(dest)

pickle.dump(clf, open(os.path.join(dest, 'best_classifier.pkl'), 'wb'), protocol=4)
print('머신러닝 데이터 저장 완료')
