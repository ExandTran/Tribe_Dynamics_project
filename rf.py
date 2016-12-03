import json
import numpy as np
import random
from collections import defaultdict
from load import data,labels,test,class0,class1
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


#opens Word2Vec dataset transforms it into a usable object
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

#Pipeline
pipeline = Pipeline([
("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
("Random Forest", RandomForestClassifier(n_jobs = 1000))])

#Creates a random subset of the data and trains it

base_length = len(class1)
best_overall_score = 0
best_overall_model = None
best_iteration = 0
for multiplier in [2]:
    for iteration in range(5):
        print("doing iteration", iteration - 1, "of 5")
        class0_subset_indices = random.sample(range(len(class0)), multiplier*len(class1))
        class0_subset = np.array([data[i] for i in class0 if i in class0_subset_indices])
        training_data = np.concatenate([class0_subset, class1])
        training_labels = np.array(([False]*len(class0_subset)) + ([True]*len(class1)))

        print("Created random training set for multiplier:", multiplier)

        test_indices = random.sample(range(len(training_data)), len(training_data) / 10)
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        for i in range(len(training_data)):
            if i in test_indices:
                test_data.append(training_data[i])
                test_labels.append(training_labels[i])
            else:
                train_data.append(training_data[i])
                train_labels.append(training_labels[i])

        print("Created random test set")

        model = pipeline.fit(train_data, train_labels)

        print("Created model")

        predictions = model.predict(test_data)

        score = precision_score(test_labels, predictions, average = 'weighted')
        spc = precision_score(test_labels, predictions, average = None)
        print("score:", score)
        print("score per class", spc)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_model = model
            best_iteration = multiplier

#writes json file
data = {"response" :best_overall_model.predict(test).tolist()}
with open('predict_rf.json', 'w') as json_file:
    json.dump(data, json_file)