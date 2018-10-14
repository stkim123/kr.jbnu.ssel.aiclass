from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pickle
import os
from time import time
import pandas as pd
from chap8.mylib.tokenizer import tokenizer, tokenizer_porter

df = pd.read_csv('./data/refined_movie_review.csv')
x_train = df.loc[:35000, 'review'].values
y_train = df.loc[:35000, 'sentiment'].values
x_test = df.loc[35000:, 'review'].values
y_test = df.loc[35000:, 'sentiment'].values

tfidf = TfidfVectorizer()
lr = LogisticRegression(C=10.0, penalty = 'l2', random_state=0)

lr_tfidf = Pipeline([('vect', tfidf), ('clf', lr)])

stime = time()
print('머신러닝 시작')
lr_tfidf.fit(x_train, y_train)
print('머신러닝 종료')

y_pred = lr_tfidf.predict(x_test)
print('테스트 종료: 소요시간 [%d]초' % (time() - stime))
print('정확도: %.3f' % accuracy_score(y_test, y_pred))

curDir = os.getcwd()
dest = os.path.join(curDir, 'classifier')
if not os.path.exists(dest) :
    os.makedirs(dest)

pickle.dump(lr_tfidf, open(os.path.join(dest, 'classifier.pkl'), 'wb'), protocol=4)
print('머신러닝 데이터 저장 완료')

