import numpy as np
from collections import defaultdict
from load import X_train,Y_train,X_test,Y_test
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


with open("glove.6B.50d.txt", "rb") as lines:
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
overall_best_score = 0
overall_best_model = None
scores = []
for i in range(-6,7):
    C = 10**i
    print("fitting C = ", C)
    pipeline = Pipeline([
    ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
    ("logistic regression", LogisticRegression(C = C))])
    model = pipeline.fit(X_train, Y_train)
    score = model.score(X_test, Y_test)
    scores.append(score)
    if score > overall_best_score:
        overall_best_score = score
        overall_best_model = model

print(scores)
print("")
print(overall_best_score)
joblib.dump(overall_best_model, 'log_reg_model.pkl') 
