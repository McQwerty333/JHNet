import numpy as np


# np.random.seed(0)


def cce_prime(outputs, true_outputs):
	output_gradient = outputs - true_outputs
	return np.array(output_gradient)


def categorical_cross_entropy(outputs, true_outputs):
	# true outputs as batch of one-hot-encoded vectors
	clipped_outputs = np.clip(outputs, 1e-7, 1 - 1e-7)
	differences = np.sum(clipped_outputs * true_outputs, axis=1)
	cce = -np.log(differences)
	return cce


def mse_prime(outputs, true_outputs):
	output_gradient = 2 / np.size(true_outputs) * (outputs - true_outputs)
	return np.array(output_gradient)


def mean_squared_error(outputs, true_outputs):
	differences = np.power(true_outputs - outputs, 2)
	sum_differences = np.sum(differences)
	mse = sum_differences / np.size(true_outputs)
	return mse


class Layer:
	def __init__(self):
		self.inputs = None
		self.outputs = None

	def propagate_forward(self, inputs):
		pass

	def propagate_backward(self, output_gradient, learning_rate):
		pass


class Dense(Layer):
	def __init__(self, feature_count, output_count):
		self.weights = .1 * np.random.randn(feature_count, output_count)
		self.biases = np.random.randn(1, output_count)

	def propagate_forward(self, inputs):
		self.inputs = inputs
		self.outputs = np.dot(inputs, self.weights) + self.biases
		return self.outputs

	def propagate_backward(self, output_gradient, learning_rate):
		self.weights -= learning_rate * np.dot(self.inputs.T, output_gradient)
		self.biases -= np.sum(learning_rate * output_gradient, axis=0, keepdims=True)
		input_gradient = np.dot(output_gradient, self.weights.T)
		return input_gradient


class Activation(Layer):
	def forward_sigmoid(self, inputs):
		self.outputs = 1 / (1 + np.exp(-inputs))
		return self.outputs

	def forward_softmax(self, inputs):
		exponent_values_normalized = np.exp(inputs - np.max(inputs, axis=0, keepdims=True))
		self.outputs = exponent_values_normalized / np.sum(exponent_values_normalized, axis=0, keepdims=True)
		return self.outputs

	def forward_relu(self, inputs):
		self.outputs = np.maximum(0, inputs)
		return self.outputs


class Network:
	def __init__(self, layer_list):
		#  like [2, 4, 4, 3]
		self.layers = []
		self.dense = []
		for layer_pair_one, layer_pair_two in zip(layer_list, layer_list[1:]):
			self.layers.append(Dense(layer_pair_one, layer_pair_two))
		self.activate = Activation()

	def train(self, training_X, training_Y, epochs, learning_rate):
		for iteration in range(epochs):
			gradient = mse_prime(self.forward(training_X), training_Y)
			self.backward(gradient, learning_rate)
			if iteration % 500 == 0:
				print(
					f'iteration {iteration} loss on training: {mean_squared_error(self.forward(training_X), training_Y)}')

	def forward(self, inputs):
		for layer in self.layers:
			inputs = self.activate.forward_sigmoid(layer.propagate_forward(inputs))
		return inputs

	def backward(self, output_gradient, learning_rate):
		for layer in reversed(self.layers):
			output_gradient = layer.propagate_backward(output_gradient, learning_rate)


if __name__ == '__main__':
	# XOR test
	layer1 = Dense(2, 1)
	activation = Activation()
	X_XOR = np.array([[0, 1],
					  [1, 0],
					  [0, 0],
					  [1, 1]])
	Y_XOR = np.array([[1], [1], [0], [1]])

	network01 = Network([2, 4, 1])
	network01.train(X_XOR, Y_XOR, 5000, .5)
	print(network01.forward(X_XOR))
