import numpy as np


# successful gradient descent of a single layer


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def softmax(inputs):
	exponent_values_normalized = np.exp(inputs - np.max(inputs, axis=0, keepdims=True))
	return exponent_values_normalized / np.sum(exponent_values_normalized, axis=0, keepdims=True)


class DenseLayer:
	def __init__(self, n_features, n_outputs):
		self.weights = .1 * np.random.randn(n_features, n_outputs)
		self.biases = np.random.randn(1, n_outputs)

	def feed_forward_sigmoid(self, inputs):
		self.inputs = inputs
		self.outputs = sigmoid(np.dot(self.inputs, self.weights) + self.biases)
		return self.outputs

	def feed_forward_softmax(self, inputs):
		self.inputs = inputs
		self.outputs = softmax(np.dot(self.inputs, self.weights) + self.biases)
		return self.outputs

	def propagate_backward(self, output_gradient, learning_rate):
		weights_gradient = np.dot(self.inputs.T, output_gradient)
		input_gradient = np.dot(output_gradient, self.weights.T)
		self.weights -= weights_gradient * learning_rate
		self.biases -= np.sum(np.array(output_gradient) * learning_rate, axis=0, keepdims=True)
		return input_gradient

	def mse(self, true_outputs):
		differences = np.power(true_outputs - self.outputs, 2)
		sum_differences = np.sum(differences)
		mean_squared_error = sum_differences / np.size(true_outputs)
		return mean_squared_error

	def mse_prime(self, true_outputs):
		output_gradient = 2 / np.size(true_outputs) * (self.outputs - true_outputs)
		return output_gradient

	def categorical_cross_entropy(self, true_outputs):
		# true outputs as batch of one-hot-encoded vectors
		clipped_outputs = np.clip(self.outputs, 1e-7, 1 - 1e-7)
		differences = np.sum(clipped_outputs * true_outputs, axis=1)
		cce = -np.log(differences)
		return cce

	def cce_prime(self, true_outputs):
		output_gradient = self.outputs - true_outputs
		return output_gradient


if __name__ == '__main__':
	layer1 = DenseLayer(2, 1)
	X_XOR = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
	Y_XOR = np.array([[1], [1], [0], [1]])

	X_OR = np.array([[0, 0], [0, 1], [1, 0]])
	Y_OR = np.array([[0], [1], [1]])

	X_AND = np.array([[0, 0], [0, 1], [1, 1]])
	Y_AND = np.array([[0], [0], [1]])

	for i in range(5000):
		output = layer1.feed_forward_sigmoid(X_XOR)
		layer1.propagate_backward(layer1.mse_prime(Y_XOR), .5)
	print(layer1.feed_forward_sigmoid(X_XOR))

	for i in range(5000):
		output = layer1.feed_forward_sigmoid(X_OR)
		layer1.propagate_backward(layer1.mse_prime(Y_OR), .5)
	print(layer1.feed_forward_sigmoid(np.array([[1, 1]])))

	for i in range(5000):
		output = layer1.feed_forward_softmax(X_AND)
		layer1.propagate_backward(layer1.cce_prime(Y_AND), .1)
	print(layer1.feed_forward_softmax(np.array([[1, 1], [0, 0], [1, 0]])))
