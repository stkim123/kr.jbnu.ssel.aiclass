
import pandas as pd
import os
import numpy as np

from chap8.mylib.progbar import ProgBar

path = './data/'

pbar = ProgBar(50000)
labels = {'pos': 1, 'neg': 0}
df = pd.DataFrame()

for s in ('test', 'train'):
    for name in ('pos','neg'):
        subpath = '%s/%s' % (s, name)
        dirpath = path + subpath
        for file in os.listdir(dirpath):
            with open(os.path.join(dirpath, file), 'r', encoding='UTF8') as f:
                txt = f.read()
            df = df.append([[txt, labels[name]]], ignore_index=True)
            pbar.update()

df.columns = ['review', 'sentiment']

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('./data/movie_review.csv', index=False)

import pandas as pd
df = pd.DataFrame()
df = pd.read_csv('./data/movie_review.csv')
print(df.head())
print(df.tail())


