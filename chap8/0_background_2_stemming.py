
from chap8.mylib.tokenizer import tokenizer, tokenizer_porter

# import nltk
# nltk.download('stopwords')

text = 'runners like running and thus they run'
print(tokenizer(text))
print(tokenizer_porter(text))

