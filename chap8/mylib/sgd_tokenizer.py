import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# import nltk
# nltk.download('stopwords')

porter = PorterStemmer()
stop = stopwords.words('english')

def sgd_tokenizer(text) :
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)|\^.?\^', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized
