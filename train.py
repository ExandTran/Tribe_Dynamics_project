import json
import numpy as np
import random
import re


with open("data/train_data.json", "r") as json_file:
    loaded_data = json.load(json_file)

longest_text = 1480

data, labels = [], []
for row in loaded_data:
	text = [""]*longest_text
	text_data = re.sub(r'[^\w\s]','',row['text']).split()
	
	for i in range(len(text_data)):
		text[i] = text_data[i]
	
	data.append([row['brand_id']] + text + [row['emv']])
	labels.append(row['feedback_bool'])

test_indices = random.sample(range(len(data)), len(data)/10)

X_train = []
X_test = []
Y_train = []
Y_test = []

for i in range(len(data)):
	if i in test_indices:
		X_test.append(data[i])
		Y_test.append(labels[i])
	else:
		X_train.append(data[i])
		Y_train.append(labels[i])
print(len(X_train), len(X_test), len(Y_train), len(Y_test))

X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)