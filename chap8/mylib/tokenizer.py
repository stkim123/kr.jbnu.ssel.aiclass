from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

porter = PorterStemmer()
stop = stopwords.words('english')

def tokenizer(text) :
    return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
