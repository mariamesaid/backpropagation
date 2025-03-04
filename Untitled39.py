#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the neural network structure
class NeuralNetwork:
    def __init__(self):
        # Initialize weights and biases
        self.weights = {
            'w1': np.random.randn(),
            'w2': np.random.randn(),
            'w3': np.random.randn(),
            'w4': np.random.randn(),
            'w5': np.random.randn(),
            'w6': np.random.randn(),
            'w7': np.random.randn(),
            'w8': np.random.randn()
        }
        self.biases = {
            'b1': np.random.randn(),
            'b2': np.random.randn(),
            'b3': np.random.randn(),
            'b4': np.random.randn()
        }

    def forward(self, x1, x2):
        # Hidden layer calculations
        h1 = sigmoid(self.weights['w1'] * x1 + self.weights['w2'] * x2 + self.biases['b1'])
        h2 = sigmoid(self.weights['w3'] * x1 + self.weights['w4'] * x2 + self.biases['b2'])

        # Output layer calculations
        o1 = sigmoid(self.weights['w5'] * h1 + self.weights['w7'] * h2 + self.biases['b3'])
        o2 = sigmoid(self.weights['w6'] * h1 + self.weights['w8'] * h2 + self.biases['b4'])

        return o1, o2

# Example usage
nn = NeuralNetwork()
x1, x2 = 0.5, 0.1  # Example input values
output1, output2 = nn.forward(x1, x2)
print(f"Output 1: {output1}, Output 2: {output2}")


# In[ ]:





# In[ ]:




