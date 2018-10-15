import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# import nltk
# nltk.download('stopwords')
from sklearn.feature_extraction.text import HashingVectorizer

porter = PorterStemmer()
stop = stopwords.words('english')

def sgd_tokenizer(text) :
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)|\^.?\^', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

vect = HashingVectorizer(decode_error='ignore', n_features=2**21,
                         preprocessor=None, tokenizer=sgd_tokenizer)

