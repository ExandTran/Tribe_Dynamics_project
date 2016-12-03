import json
import numpy as np
import textmining


with open("data/train_data.json", "r") as json_file:
    loaded_data = json.load(json_file)

with open("data/test_data_no_response.json", "r") as json_file:
    test_data = json.load(json_file)

print("loaded data")

data, labels = [], []
for row in loaded_data:
    data.append([row['brand_id'],row['text']])
    labels.append(row['feedback_bool'])

test = []
for row in test_data:
    test.append([row['brand_id'],row['text']])

print("Converted data to usable objects")
print(data[0])
print(test[0])
tdm = textmining.TermDocumentMatrix()
data_obs = len(data)
test_obs = len(test)
total = data + test
for i in total:
    tdm.add_doc(i[1])

data = []
test = []
count = 0
for row in tdm.rows():
    if count == 0:
        pass
    if count >= data_obs:
        test.append([total[count - 1][0]] + row)
    else:
        data.append([total[count - 1][0]] + row)
    count += 1

print("created train and test TDM")

class0 = []
class1 = []
for i in range(len(labels)):
    if labels[i] == False:
        class0.append(i)
    else:
        class1.append(i)
class1 = np.array([data[i] for i in class1])





