from nltk.tokenize import WordPunctTokenizer
from nltk.stem import PorterStemmer

from nltk.corpus import stopwords

tokenizer = WordPunctTokenizer()
stemmer = PorterStemmer()
stop_words_list = stopwords.words("english")

def preprocess(text):
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    stems = [stemmer.stem(token) for token in tokens]
    filtered_tokens = [x for x in stems if x not in stop_words_list]
    return filtered_tokens