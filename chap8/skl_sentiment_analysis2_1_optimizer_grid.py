

import pickle
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

import datetime
from time import time

from sent_analysis2.mylib.tokenizer import tokenizer, tokenizer_porter
from nltk.corpus import stopwords

if __name__ == "__main__" :
    stop = stopwords.words('english')
    df = pd.read_csv('./data/refined_movie_review.csv')
    x_train = df.loc[:35000, 'review'].values
    y_train = df.loc[:35000, 'sentiment'].values
    x_test = df.loc[35000:, 'review'].values
    y_test = df.loc[35000:, 'sentiment'].values

    tfidf = TfidfVectorizer(lowercase=False)

    param_grid = [{'vect__ngram_range':[(1,1)], 'vect__stop_words':[stop, None],
                   'vect__tokenizer' : [tokenizer, tokenizer_porter],
                   'clf__penalty' : ['l1', 'l2'], 'clf__C': [1.0, 10.0, 100.0]},
                   {'vect__ngram_range':[(1,1)], 'vect__stop_words':[stop, None],
                   'vect__tokenizer' : [tokenizer, tokenizer_porter],
                   'vect__use_idf' : [False],'vect__norm' : [None],
                   'clf__penalty' : ['l1', 'l2'], 'clf__C': [1.0, 10.0, 100.0]}]

    lr = LogisticRegression(random_state=0)

    lr_tfidf = Pipeline([('vect', tfidf), ('clf', lr)])
    gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)


    print('머신러닝 최적 파라미터 계산 시작: %s' % (datetime.datetime.now()))
    gs_lr_tfidf.fit(x_train, y_train)
    print('머신러닝 최적 파라미터 계산 종료: %s' % (datetime.datetime.now()))
    print(gs_lr_tfidf.best_params_)

    clf=gs_lr_tfidf.best_estimator_
    print('테스트 정확도: %.3f' % clf.score(x_test, y_test))
    y_pred = lr_tfidf.predict(x_test)
    print('테스트 종료: 소요시간 [%d]초' % (time() - stime))
    print('정확도: %.3f' % accuracy_score(y_test, y_pred))

    curDir = os.getcwd()
    dest = os.path.join(curDir, 'classifier')
    if not os.path.exists(dest) :
        os.makedirs(dest)

    pickle.dump(clf, open(os.path.join(dest, 'best_classifier.pkl'), 'wb'), protocol=4)
    print('머신러닝 데이터 저장 완료')

