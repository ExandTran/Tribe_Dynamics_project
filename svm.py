import json
import numpy as np
from collections import defaultdict
from load import data,labels,test
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

with open("glove.6B.300d.txt", "rb") as lines:
    w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
           for line in lines}
print("finished reading word2vec...")

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

pipeline = Pipeline([
("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
("radial svm", svm.SVC(kernel = "rbf", C=100))])
data = {"response" :pipeline.fit(data, labels).predict(test).tolist()}
with open('predict.json', 'w') as json_file:
    json.dump(data, json_file)

# joblib.dump(overall_best_model, 'svm_model.pkl') 