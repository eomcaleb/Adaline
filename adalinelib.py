import numpy as np

class Adaline(object):
	def __init__(self, ninput, epochs = 40, learning_rate=0.01, random_weights = True):
		self.epochs = epochs
		self.learning_rate = learning_rate
		self.ninput = ninput
		if (random_weights):
			self.weights = np.zeros(ninput + 1) # adds 1 is for the bias at the 0th position
		else:
			self.weights = np.random.rand(ninput + 1)

	# linearfunction - used for training the model to get the output/sum
	def linearfunction(self, inputs):
		summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
		return summation	

	# predict - used after training with real life data set
	def predict(self, inputs):
		if (np.dot(inputs, self.weights[1:]) + self.weights[0]) > 0.0:
			return 1
		return -1

	# train - explicit programming the training procedure
	def train(self, trainingdata, targets, verbose = False):
		self.cost = []
		for epoch in range(self.epochs): 
			for inputs, target in zip(trainingdata, targets):
				predict = self.linearfunction(inputs)
				# Stochastic Gradient Descent
				self.weights[1:] += self.learning_rate * (target - predict) * inputs
				self.weights[0] += self.learning_rate * (target - predict)
				# Keep track of cost 
				self.cost.append(np.sum((target - predict) ** 2)/ self.ninput)
			if (verbose and epoch % 10 == 0):
				print ("Epoch #", epoch, "; Bias: ", self.weights[0], "; Weights:", self.weights[1:], "; Error = ", (np.sum((target - predict) ** 2)/ self.ninput))	
	