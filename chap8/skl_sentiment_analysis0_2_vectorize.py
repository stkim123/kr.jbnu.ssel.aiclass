
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import numpy as np

txt1 = 'the car is expensive'
txt2 = 'the truck is cheap'
txt3 = 'the car is expensive and the truck is cheap'

count = CountVectorizer()
docs = np.array([txt1, txt2, txt3])
bag = count.fit_transform(docs)

print(count.vocabulary_)
print(bag.toarray())

tfidf = TfidfTransformer()
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())