import json
import numpy as np
import random
from collections import defaultdict
from load2 import data,labels,test,class0,class1
from sklearn import svm
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

base_length = len(class1)
best_overall_score = 0
best_overall_model = None
best_iteration = 0
clf = svm.SVC(kernel = 'radial', C = 100)
for multiplier in [1,2,3]:
    for iteration in range(5):
        print("doing iteration", iteration, "of 5")
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

        model = clf.fit(train_data, train_labels)

        print("Created model")

        predictions = model.predict(test_data)

        score = precision_score(test_labels, predictions)
        spc = precision_score(test_labels, predictions, average = None)
        print("score:", score)
        print("score per class", spc)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_model = model
            best_iteration = multiplier

#writes json file
data = {"response" :best_overall_model.predict(test).tolist()}
with open('predict_reg_svm.json', 'w') as json_file:
    json.dump(data, json_file)