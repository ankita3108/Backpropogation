# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:14:57 2019

@author: agupta466
"""
from random import random
from random import seed
from math import exp

## Initialize a network
def initialize_network(n_inputs, n_hidden, hidden_layers, n_outputs):
    network = list()
    
    ## Connecting Input to first hidden layer
    hidden_layer1 = [{"weights" : [random() for i in range(n_inputs)]} for i in range(n_hidden)]
    network.append(hidden_layer1)
    
    ## Adding more hidden layers to the network
    for i in range(hidden_layers - 1):
        hidden_layer = [{"weights" : [random() for i in range(n_hidden)]} for i in range(n_hidden)]
        network.append(hidden_layer)
    
    ## Adding an output layer
    output_layer = [{"weights" : [random() for i in range(n_hidden)]} for i in range(n_outputs)]
    network.append(output_layer)
    return(network)

def generate_input(n_inputs):
    input_value = [random() for i in range(n_inputs)]
    return(input_value)

## Calculate activation neurons for the input
def activate(weights, inputs):   
    ## Bias is assumed to be the last weight, assigning last weight as bias
    activation = weights[-1]
    
    ## Calculating activation for each neuron in hidden layers
    for i in range(len(weights) - 1):
        activation+= weights[i] * inputs[i]    
    return(activation)

def transfer(activation):
    ## Adding a transfer function (sigmoid) to the activation
    transfer_func = 1/(1+exp(-activation))
    return(transfer_func)

def forward_propogate(network, inputs):
    
    ## Calculating activations for all neurons in a layer
    for layer in network:
        new_inputs = []
        
    ## Calculating activation for each neuron in a layer and feeding as input to next layer
        for neuron in layer:
            activation = activate(neuron["weights"], inputs)
            neuron["output"] = transfer(activation)
            new_inputs.append(neuron["output"])
        inputs = new_inputs
    return(inputs)

## Calculate the derivative of neuron output
def transfer_derivative(output):
    
    ## derivative = S(x) * (1 - S(x))
    derivative = output * (1 - output)
    return(derivative)
    
def backward_propogate(network, expected):
    
    ## Start from the last layer in network
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        
        ## This code will calculate delta error in each neuron for each layer that will propogate in hidden layers
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i+1]:
                    error += (neuron["weights"][j] * neuron["delta"])
                errors.append(error)
        
        ## This code will calculate error in output layer
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                error = (expected[j] - neuron["output"])
                errors.append(error)
                
    ## This code will calculate the cost derivative that will be propagate through layers
    ## dE(Total)/dwi = dE(Total)/d(Hi) * d(Hi(out))/d(Hi(input)) * d(Hi(input))/dwi
    ## dE(Total)/dwi = (Output - Target) * output (1 - output) * Hi(output)
        
        for j in range(len(layer)):
            neuron = layer[j]
            neuron["delta"] = errors[j] * transfer_derivative(neuron["output"])
            
    return(network)

## Code to update weights after backpropogation    

def update_weights(network, inputs, l_rate):
    ## Updating weights - starting from 1st layer
    for i in range(len(network)):
        ## Getting all inputs except last one (bias)
        network_input = inputs[:-1]
        if i!= 0:
            ## Getting Hi(output) for each successive layer
            network_input = [neuron["output"] for neuron in network[i - 1]]
        for neuron in network[i]:
            ## For each neuron -> calculating error differential w.r.t. to different weights linked to different inputs
            for j in range(len(network_input)):
                neuron["weights"][j] += l_rate * neuron["delta"] * network_input[j]
            ## Updating the last neuron with only differential (its bias)
            neuron["weights"][-1] += l_rate * neuron["delta"]

def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propogate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propogate(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

## Predicting using network
def predict(network, row):
	outputs = forward_propogate(network, row)
	return outputs.index(max(outputs))

def main():
    
    seed(100)
    
    dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
    
    n_inputs = len(dataset[0]) - 1
    n_outputs = len(set([row[-1] for row in dataset]))
    
    network = initialize_network(n_inputs, 6, 1, n_outputs)
    
    train_network(network, dataset, 0.5, 20, n_outputs)
    
    for row in dataset:
        prediction = predict(network, row)
        print('Expected=%d, Got=%d' % (row[-1], prediction))    

if __name__ == '__main__':
    main()