import numpy as np
from pandas_03 import X_train, Y_train_OHE, X_val, Y_train, Y_val


def mean_squared_error(outputs, true_outputs):
	differences = np.power(true_outputs - outputs, 2)
	sum_differences = np.sum(differences)
	mse = sum_differences / np.size(true_outputs)
	return mse


def categorical_cross_entropy(outputs, true_outputs):
	# true outputs as batch of one-hot-encoded vectors
	clipped_outputs = np.clip(outputs, 1e-7, 1 - 1e-7)
	differences = np.sum(clipped_outputs * true_outputs, axis=1)
	cce = -np.log(differences)
	return cce


class Layer:
	def __init__(self):
		self.inputs = None
		self.outputs = None

	def forward(self, inputs):
		pass

	def backward(self, output_gradient, learning_rate):
		pass


class Gradient:
	def gradient(self, outputs, true_outputs):
		pass


class Dense(Layer):
	def __init__(self, feature_count, output_count):
		self.weights = .1 * np.random.randn(feature_count, output_count)
		self.biases = np.random.randn(1, output_count)

	def forward(self, inputs):
		self.inputs = inputs
		self.outputs = np.dot(inputs, self.weights) + self.biases
		return self.outputs

	def backward(self, output_gradient, learning_rate):
		self.weights -= learning_rate * np.dot(self.inputs.T, output_gradient)
		self.biases -= np.sum(learning_rate * output_gradient, axis=0, keepdims=True)
		input_gradient = np.dot(output_gradient, self.weights.T)
		return input_gradient


class SigmoidActivation(Layer):
	def forward(self, inputs):
		return 1 / (np.exp(-inputs) + 1)


class SoftmaxActivation(Layer):
	def forward(self, inputs):
		exp = np.exp(inputs - inputs.max())
		return exp / np.sum(exp, axis=1, keepdims=True)


class MsePrime(Gradient):
	def gradient(self, outputs, true_outputs):
		output_gradient = 2 / np.size(true_outputs) * (outputs - true_outputs)
		return np.array(output_gradient)


class CcePrime(Gradient):
	def gradient(self, outputs, true_outputs):
		return np.array(outputs - true_outputs)


class Network:
	def __init__(self, layer_list, activation, output_activation, gradient_function):
		self.layers = []
		for n_in, n_out in zip(layer_list, layer_list[1:]):
			self.layers.append(Dense(n_in, n_out))
		self.activation = activation
		self.outputActivation = output_activation
		self.gradient_function = gradient_function

	def forward(self, inputs):
		for layer in self.layers[:-1]:
			inputs = self.activation.forward(layer.forward(inputs))
		inputs = self.outputActivation.forward(self.layers[-1].forward(inputs))
		return inputs

	def backward(self, output_gradient, learning_rate):
		for layer in reversed(self.layers):
			output_gradient = layer.backward(output_gradient, learning_rate)

	def train(self, training_X, training_Y, epochs, learning_rate):
		for iteration in range(epochs):
			gradient = self.gradient_function.gradient(self.forward(training_X), training_Y)
			self.backward(gradient, learning_rate)


def train_batch_gd(network, iterations, learning_rate):
	print("\nbatch results")
	print(f"parameters: {iterations} iterations, {learning_rate} learning rate")
	network.train(X_train, Y_train_OHE, iterations, learning_rate)
	predicted = np.argmax(network_01.forward(X_train), axis=1)
	print("training accuracy: ", (predicted == Y_train).sum() / predicted.size)
	predicted = np.argmax(network_01.forward(X_val), axis=1)
	print("validation accuracy: ", (predicted == Y_val).sum() / predicted.size)
	print(predicted[:20], "\n", Y_val[:20])


def train_minibatch_gd(network, batch_size, iterations, learning_rate):
	print("\nminibatch results:")
	print(f"parameters: {batch_size} batch size, {iterations} iterations, {learning_rate} learning rate")
	for i in range(50000 // batch_size):
		network.train(X_train[batch_size * i: batch_size * (i + 1), :],
					  Y_train_OHE[batch_size * i: batch_size * (i + 1), :], iterations, learning_rate)
	predicted = np.argmax(network_01.forward(X_train), axis=1)
	print("training accuracy: ", (predicted == Y_train).sum() / predicted.size)
	predicted = np.argmax(network_01.forward(X_val), axis=1)
	print("validation accuracy: ", (predicted == Y_val).sum() / predicted.size)
	print(predicted[:20], "\n", Y_val[:20])


def train_stochastic_gd(network, iterations, learning_rate):
	print("\nstochastic results: ")
	print(f"parameters: {iterations} iterations, {learning_rate} learning rate")
	for i in range(49999):
		network.train(X_train[i:i + 1, :], Y_train_OHE[i:i + 1, :], iterations, learning_rate)
	predicted = np.argmax(network_01.forward(X_train), axis=1)
	print("training accuracy: ", (predicted == Y_train).sum() / predicted.size)
	predicted = np.argmax(network_01.forward(X_val), axis=1)
	print("validation accuracy: ", (predicted == Y_val).sum() / predicted.size)
	print(predicted[:20], "\n", Y_val[:20])


if __name__ == '__main__':
	sigmoid_activation = SigmoidActivation()
	softmax_activation = SoftmaxActivation()
	mse_prime = MsePrime()
	cce_prime = CcePrime()

	network_01 = Network([784, 16, 16, 10], sigmoid_activation, softmax_activation, cce_prime)
	np.set_printoptions(precision=2, suppress=True)
	train_minibatch_gd(network_01, 8, 1, .01)
	train_stochastic_gd(network_01, 1, .01)
	train_batch_gd(network_01, 1000, .00002)
