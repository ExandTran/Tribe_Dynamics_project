import json


with open("data/train_data.json", "r") as json_file:
    loaded_data = json.load(json_file)

train_data, train_labels = [], []
for data in loaded_data:
    train_data.append(data['text'])
    train_labels.append(data['feedback_bool'])
