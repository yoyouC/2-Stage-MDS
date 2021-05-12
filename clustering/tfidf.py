from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import nltk
import re

stemmer = SnowballStemmer("english")

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    # stems = [stemmer.stem(t) for t in filtered_tokens]

    return filtered_tokens

def cluster_doc(doc_strs, n, method='kmeans'):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

    tfidf_matrix = tfidf_vectorizer.fit_transform(doc_strs) #fit the vectorizer to synopses

    if method == 'k-means':
        labels = KMeans(n_clusters=n, random_state=0).fit(tfidf_matrix).labels_
    else:
        labels = AgglomerativeClustering(n_clusters=n).fit(tfidf_matrix.toarray()).labels_


    return None, labels
    