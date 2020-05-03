import numpy as np
from adalinelib import Adaline

# Training data set
# NOTE: bi-polar input is used for ADALINE
training_inputs = []
training_inputs.append(np.array([1,1]))
training_inputs.append(np.array([1,-1]))
training_inputs.append(np.array([-1,1]))
training_inputs.append(np.array([-1,-1]))

# AND GATE
labels = np.array([1,-1,-1,-1]) 

# ADALINE
adaline = Adaline(2)
adaline.train(training_inputs, labels, True)
#adaline.train2(training_inputs, labels, True)

# Real Data Set
print ("Real Data Set")
input_1 = np.array([1,1])
print ("Input: ", input_1)
print (adaline.reallifepredict(input_1))
adaline.costfunction()
