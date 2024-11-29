import spacy
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer


# Declare Constants and Variables
NLP = spacy.load('en_core_web_sm')
STOPWORDS = stopwords.words('english')
VECTORIZER = TfidfVectorizer(stop_words=STOPWORDS)
DATASET_PATH = "data/books.csv"
NUM_CLUSTERS = range(1,40)
SSE = []
MIN_NUM_RATINGS = 2000000
CLUSTER_PARAMS = {
    "init":"random",
    "max_iter": 100,
    "n_init":10,
    "random_state":None
}
PUNCTUATION = string.punctuation + '—'