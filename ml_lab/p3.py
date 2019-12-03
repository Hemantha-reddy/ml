import numpy as np
from csv import reader
from math import log2
from collections import Counter
from pprint import pprint


class Node:
    def __init__(self, label):
        self.label = label
        self.branches = {}

def entropy(data):
    total, positive, negative = len(data), (data[:, -1] == "Y").sum(), (data[:, -1] == "N").sum()
    entropy = 0
    if positive:
        entropy -= positive / total * log2(positive / total)
    if negative:
        entropy -= negative / total * log2(negative / total)
    return entropy

def gain(s, data, column):
    values = set(data[:, column])
    gain = s
    for value in values:
        sub = data[data[:, column] == value]
        #print("val is "+value+" ",sub)
        gain -= len(sub) / len(data) * entropy(sub)
    return gain

def bestAttribute(data):
    s = entropy(data)
    g = [gain(s, data, column) for column in range(len(data[0]) - 1)]
    return g.index(max(g))

def id3(data, labels):
    #print("data is ",data)
    root = Node('Null')
    if entropy(data) == 0:
        #all are either pos or neg
        root.label = data[0, -1]

    elif len(data[0]) == 1:
        root.label = Counter(data[:, -1]).most_common()[0][0]

    else:
        column = bestAttribute(data)
        root.label = labels[column]
        values = set(data[:, column])
        #print("label is "+root.label+"  values are ",values)
        for value in values:
            nData = np.delete(
                data[data[:, column] == value], column, axis=1)
            #print("ndata ",nData)
            nLabels = np.delete(labels, column)
            root.branches[value] = id3(nData, nLabels)
    return root

def getRules(root, rule, rules):
    if not root.branches:
        #print("rule[:-2] is ",rule[:-2])
        rules.append(rule[:-2] + "=> " + root.label)
    for value, nRoot in root.branches.items():
        getRules(nRoot, rule + root.label + "=" + value + " ^ ", rules)

def predict(tree, tup):
    if not tree.branches:
        return tree.label
    return predict(tree.branches[tup[tree.label]], tup)



with open('Dataset3.csv') as f:
    data = np.array(list(reader(f)))
labels = np.array(['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis'])
tree = id3(data, labels)
rules = []
getRules(tree, "", rules)
pprint(rules)

tup = {}
for label in labels[:-1]:
    tup[label] = input(label + ": ")

pprint(predict(tree, tup))