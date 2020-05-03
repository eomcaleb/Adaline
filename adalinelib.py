import numpy as np

class Adaline(object):
	def __init__(self, ninput, epochs = 40, learning_rate=0.1, random_weights = True):
		self.epochs = epochs
		self.learning_rate = learning_rate
		self.ninput = ninput
		if (random_weights):
			self.weights = np.zeros(ninput + 1) # adds 1 is for the bias at the 0th position
		else:
			self.weights = np.random.rand(ninput + 1)

	# predict - used for training the model to get the output/sum
	def predict(self, inputs):
		summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
		return summation
	
	# reallifepredict - used after training with real life data set
	def reallifepredict(self, inputs):
		if (np.dot(inputs, self.weights[1:]) + self.weights[0]) > 0.0:
			return 1
		return -1

	# costfunction - prints cost function through each epoch
	def costfunction(self):
		print("\nCost Function")
		print("Number of epoch from last iteration = ", len(self.cost))
		print(self.cost)

	# train - explicit programming the training procedure
	def train(self, trainingdata, targets, verbose = False):
		self.cost = []
		for epoch in range(self.epochs): 
			error = []
			for inputs, target in zip(trainingdata, targets):
				predict = self.predict(inputs)
				self.weights[1:] += self.learning_rate * (target - predict) * inputs
				self.weights[0] += self.learning_rate * (target - predict)
				error.append((target - predict) ** 2)
			self.cost.append(np.sum(error) / self.ninput)
			if (verbose):
				print ("Epoch #", epoch, "; Bias: ", self.weights[0], "; Weights:", self.weights[1:])	
	
	# train2 - implicit numpy library call for training procedure (same result as train) 
	def train2(self, trainingdata, targets, verbose = False):
		self.cost = []
		trainingdata = np.copy(trainingdata)
		for epoch in range(self.epochs): 
			output = self.predict(trainingdata)
			errors = (targets - output)
			self.weights[1:] += self.learning_rate * trainingdata.T.dot(errors)
			self.weights[0] += self.learning_rate * errors.sum()
			self.cost.append((errors ** 2).sum() / self.ninput)
			if (verbose):
				print ("Epoch #", epoch, "; Bias: ", self.weights[0], "; Weights:", self.weights[1:])	