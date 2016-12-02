import json

"""
Compares the amount of train labels to test labels
"""

with open("data/train_data.json", "r") as json_file:
    loaded_data = json.load(json_file)

labels = []
for row in loaded_data:
	labels.append(row['feedback_bool'])

with open('predict.json', 'r') as json_file:
	predict = json.load(json_file)

true = [i for i in labels if i == True]
truths = [i for i in predict['response'] if i == True]
print("train", len(labels), len(true))
print("test", len(predict['response']), len(truths))