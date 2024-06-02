import csv
import random

import numpy as numpy


class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        self.class_counts = {}
        self.attribute_counts = {}
        self.instance_count = 0
        self.alpha = alpha

    def train(self, data):
        for instance in data:
            label = instance[-1]
            self.class_counts[label] = self.class_counts.get(label, 0) + 1

            if label not in self.attribute_counts:
                self.attribute_counts[label] = []

            for i, attribute_value in enumerate(instance[:-1]):
                if i >= len(self.attribute_counts[label]):
                    self.attribute_counts[label].append({})

                self.attribute_counts[label][i][attribute_value] = self.attribute_counts[label][i].get(attribute_value, 0) + 1

            self.instance_count += 1

    def predict(self, instance):
        max_prob = -1
        best_label = None

        for label, label_count in self.class_counts.items():
            prob = label_count / self.instance_count

            for i, attribute_value in enumerate(instance[:-1]):
                attribute_value_counts = self.attribute_counts[label][i]
                attribute_value_count = attribute_value_counts.get(attribute_value, 0)
                prob *= (attribute_value_count + self.alpha) / (label_count + self.alpha * len(attribute_value_counts))

            if prob > max_prob:
                max_prob = prob
                best_label = label

        return best_label

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
    return data

def split_data(data, train_ratio=0.7, test_ratio=0.3):
    if train_ratio + test_ratio > 1.0:
        raise ValueError("Sum of train_ratio and test_ratio should not exceed 1.0")

    random.shuffle(data)
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_size = int(len(data) * test_ratio)
    test_data = data[train_size:train_size+test_size]
    return train_data, test_data

def accuracy(test_data, predictions):
    correct = sum(1 for i, instance in enumerate(test_data) if instance[-1] == predictions[i])
    return correct / len(test_data)

data = load_data("data/car_evaluation.data")

acc_table = []
round_number = 10
for i in range(0,round_number):
    train_data, test_data = split_data(data)

    classifier = NaiveBayesClassifier()
    classifier.train(train_data)

    predictions = [classifier.predict(instance) for instance in test_data]

    acc = accuracy(test_data, predictions)
    acc_table.append(acc)

    print(f'Accuracy round {i+1}: {acc:.4f}')

print(f"Average Accuracy from {round_number} rounds: {numpy.mean(acc_table)}")
print(f"Standard deviation of Accuracy from {round_number} rounds: {numpy.std(acc_table)}")