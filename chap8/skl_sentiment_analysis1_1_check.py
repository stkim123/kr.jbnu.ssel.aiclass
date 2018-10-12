
import pandas as pd

df = pd.DataFrame()
df = pd.read_csv('./data/refined_movie_review.csv')

print(df.head())
print(df.tail())
