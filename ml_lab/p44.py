import csv
from math import exp 
from random import seed 
from random import random 

def initialize_network(n_inputs, n_hidden, n_outputs): 
	network = list() 
	hidden_layer = [{'Weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)] 
	network.append(hidden_layer) 
	output_layer = [{'Weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)] 
	network.append(output_layer) 
	return network 

def activate(Weights, inputs): 
	activation = Weights[-1] 
	for i in range(len(Weights)-1): 
		activation += Weights[i] * inputs[i] 
	return activation 

def transfer(activation): 
	return 1.0 / (1.0 + exp(-activation)) 

def forward_propagate(network, row): 
	inputs = row 
	for layer in network: 
		new_inputs = [] 
		for neuron in layer: 
			activation = activate(neuron['Weights'], inputs)
			neuron['Output'] = transfer(activation) 
			new_inputs.append(neuron['Output']) 
		inputs = new_inputs 
	return inputs 

def transfer_derivative(output): 
	return output * (1.0 - output) 

def backward_propagate_error(network, expected): 
	for i in reversed(range(len(network))): 
		layer = network[i] 
		errors = list() 
		if i != len(network)-1: 
			for j in range(len(layer)): 
				error = 0.0 
				for neuron in network[i + 1]: 
					error += (neuron['Weights'][j] * neuron['Delta']) 
				errors.append(error) 
		else: 
			for j in range(len(layer)): 
				neuron = layer[j] 
				errors.append(expected[j] - neuron['Output']) 
		for j in range(len(layer)): 
			neuron = layer[j] 
			neuron['Delta'] = errors[j] * transfer_derivative(neuron['Output']) 

def update_weights(network, row, l_rate): 
	for i in range(len(network)): 
		inputs = row[:-1] 
		if i != 0: 
			inputs = [neuron['Output'] for neuron in network[i - 1]] 
		for neuron in network[i]: 
			for j in range(len(inputs)): 
				neuron['Weights'][j] += l_rate * neuron['Delta'] * inputs[j] 
			neuron['Weights'][-1] += l_rate * neuron['Delta'] 

def train_network(network, train, l_rate, n_epoch, n_outputs): 
	for epoch in range(n_epoch): 
		sum_error = 0 
		for row in train: 
			outputs = forward_propagate(network, row) 
			expected = [0 for i in range(n_outputs)] 
			expected[int(row[-1])] = 1 
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))]) 
			backward_propagate_error(network, expected) 
			update_weights(network, row, l_rate) 
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error)) 

def loadCsv(filename):
	lines = csv.reader(open(filename))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

filename='Dataset4.csv'
dataset = loadCsv(filename)
print("Dataset:")
print(dataset)

n_inputs = len(dataset[0]) - 1 
n_outputs = len(set([row[-1] for row in dataset])) 
network = initialize_network(n_inputs, 2, n_outputs) 
train_network(network, dataset, 0.5, int(input("\nEnter no. of epochs: ")), n_outputs)
for layer in network: 
	print("Layer:")
	for i in layer:
		print('Output:', i['Output'], '\tWeights:', i['Weights'], '\tDelta', i['Delta'])
	print()

