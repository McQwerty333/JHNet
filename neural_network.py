import numpy as np


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
	return x * (1 - x)


class NeuralNetwork:
	def __init__(self, count):
		self.weights = np.random.random((count, 1)) * 2 - 1

	def train(self, training_inputs, training_outputs, iterations):
		for i in range(iterations):
			outputs = self.forward(training_inputs)
			error = training_outputs - outputs
			adjustments = error * sigmoid_prime(outputs)
			self.weights += np.dot(training_inputs.T, adjustments)

	def forward(self, inputs):
		inputs = inputs.astype(float)
		outputs = sigmoid(np.dot(inputs, self.weights))
		return outputs


if __name__ == "__main__":
	network_x1 = NeuralNetwork(3)
	train_in_x1 = np.array([[1, 0, 1],
							[0, 0, 1],
							[1, 1, 1],
							[0, 1, 1]])
	train_out_x1 = np.array([[1, 0, 1, 0]]).T
	network_x1.train(train_in_x1, train_out_x1, 5000)
	print("Training network on input where output is 1 iff x1 is 1")
	print("Testing with new input [1, 0, 0]")
	print("Output: ")
	print(network_x1.forward(np.array([1, 0, 0])))

	retrain_out_NOTx1 = np.array([[0, 1, 0, 1]]).T
	network_x1.train(train_in_x1, retrain_out_NOTx1, 5000)
	print("\nTest same network retrained for NOT x1")
	print("Output on [1, 0, 0]")
	print(network_x1.forward(np.array([1, 0, 0])))

	network_XOR = NeuralNetwork(2)
	train_in_XOR = np.array([[1, 0],
							 [1, 1],
							 [0, 0],
							 [0, 1]])
	train_out_XOR = np.array([[1, 0, 0, 1]]).T
	network_XOR.train(train_in_XOR, train_out_XOR, 50000)
	print("\nTest XOR network on input [0, 1]")
	print(network_XOR.forward(np.array([0, 1])))
	print("fails on problem with non-linear decision boundary")

	network_OR = NeuralNetwork(2)
	train_in_OR = np.array([[0, 0],
							[0, 1],
							[1, 0]])
	train_out_OR = np.array([[0, 1, 1]]).T
	network_OR.train(train_in_OR, train_out_OR, 5000)
	print("\nTest OR on new input [1, 1]")
	print(network_OR.forward(np.array([1, 1])))
	print("works with linear decision boundary problem OR")
