import numpy as np
from adalinelib import Adaline

# Training data set
training_inputs = []
training_inputs.append(np.array([1,1]))
training_inputs.append(np.array([1,-1]))
training_inputs.append(np.array([-1,1]))
training_inputs.append(np.array([-1,-1]))

# OR GATE
labels = np.array([1,1,1,-1]) 

# ADALINE
adaline = Adaline(2)
adaline.train(training_inputs, labels, True)

# Real Data Set
print ("Real Data Set")
input_1 = np.array([1,1])
print ("Input: ", input_1)
print (adaline.predict(input_1))