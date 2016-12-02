import json
import numpy as np
import random
import re


with open("data/train_data.json", "r") as json_file:
    loaded_data = json.load(json_file)

with open("data/test_data_no_response.json", "r") as json_file:
    test_data = json.load(json_file)

longest_text = 1946
word_to_ids = {}
num_words = 0

data, labels = [], []
id_data = []
for row in loaded_data:
    text = [""] * longest_text
    text_data = re.sub(r'[^\w\s]', "", row['text']).split()

    ids = [longest_text + 1] * longest_text
    for i in range(len(text_data)):
        text[i] = text_data[i]
        if text_data[i] not in word_to_ids:
            word_to_ids[text_data[i]] = num_words
            num_words += 1
        ids[i] = word_to_ids[text_data[i]]

    data.append([row['brand_id']] + text + [row['emv']])
    labels.append(row['feedback_bool'])
    id_data.append(ids)

test = []
for row in test_data:
    text = [""] * longest_text
    text_data = re.sub(r'[^\w\s]', "", row['text']).split()
    for i in range(len(text_data)):
        text[i] = text_data[i]
    test.append([row['brand_id']] + text + [row['emv']])

test_indices = random.sample(range(len(data)), len(data) / 10)

X_train = []
X_id_train = []
X_test = []
X_id_test = []
Y_train = []
Y_test = []

for i in range(len(data)):
    if i in test_indices:
        X_test.append(data[i])
        X_id_test.append(id_data[i])
        Y_test.append(labels[i])
    else:
        X_train.append(data[i])
        X_id_train.append(id_data[i])
        Y_train.append(labels[i])

X_train = np.array(X_train)
X_id_train = np.array(X_id_train)
X_test = np.array(X_test)
X_id_test = np.array(X_id_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)
data = np.array(data)
labels = np.array(labels)

