import numpy as np
import pandas as pd

# backpropagation with optimizers
# with tester class added to more easily test average performance of training methods


# np.random.seed(0)


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
		# glorot initialization
		variance = 2.0 / (feature_count + output_count)
		self.stddev = 4 * np.sqrt(variance)
		self.weights = np.random.normal(0.0, self.stddev, (feature_count, output_count))
		self.biases = np.random.randn(1, output_count)

		# momentum
		self.mw = np.zeros((feature_count, output_count))
		self.mb = np.zeros((1, output_count))

		# adagrad, rmsprop
		self.vw = np.zeros_like(self.weights)
		self.vb = np.zeros_like(self.biases)

		# adadelta
		self.ddw = np.zeros_like(self.weights)
		self.ddb = np.zeros_like(self.biases)

		# adam + extensions
		self.i = 1

		# amsgrad
		self.vhatw = np.zeros_like(self.weights)
		self.vhatb = np.zeros_like(self.biases)

	def reset(self):
		self.weights = np.random.normal(0.0, self.stddev, *self.weights)
		self.biases = np.random.randn(*self.biases)

		self.mw = np.zeros(*self.weights)
		self.mb = np.zeros(*self.biases)
		self.vw = np.zeros_like(self.weights)
		self.vb = np.zeros_like(self.biases)
		self.ddw = np.zeros_like(self.weights)
		self.ddb = np.zeros_like(self.biases)
		self.i = 1
		self.vhatw = np.zeros_like(self.weights)
		self.vhatb = np.zeros_like(self.biases)

	def forward(self, inputs):
		self.inputs = inputs
		self.outputs = np.dot(inputs, self.weights) + self.biases
		return self.outputs

	def backward(self, output_gradient, learning_rate):
		# always kept the same
		dw = np.dot(self.inputs.T, output_gradient)
		db = np.sum(output_gradient, axis=0, keepdims=True)

		# vanilla: .003
		self.weights -= learning_rate * dw
		self.biases -= learning_rate * db

		# momentum: .0005
		# beta = .9
		# self.mw = beta * self.mw - learning_rate * dw
		# self.mb = beta * self.mb - learning_rate * db
		# self.weights += self.mw
		# self.biases += self.mb

		# adagrad: .01
		# epsilon = 1e-7
		# self.vw += dw ** 2
		# self.vb += db ** 2
		# self.weights -= learning_rate * dw / np.sqrt(self.vw + epsilon)
		# self.biases -= learning_rate * db / np.sqrt(self.vb + epsilon)

		# rmsprop: .03
		# beta = .9
		# epsilon = 1e-7
		# self.vw = beta * self.vw + (1-beta) * dw ** 2
		# self.vb = beta * self.vb + (1-beta) * db ** 2
		# self.weights -= learning_rate * dw / np.sqrt(self.vw + epsilon)
		# self.biases -= learning_rate * db / np.sqrt(self.vb + epsilon)

		# adadelta: does not depend on learning rate
		# beta = .9
		# epsilon = 1e-7
		# self.vw = self.vw * beta + (1-beta) * dw ** 2
		# self.vb = self.vb * beta + (1-beta) * db ** 2
		# delta_w = np.sqrt(self.ddw + epsilon) * dw / np.sqrt(self.vw + epsilon)
		# delta_b = np.sqrt(self.ddb + epsilon) * db / np.sqrt(self.vb + epsilon)
		# self.ddw = beta * self.ddw + (1-beta) * delta_w ** 2
		# self.ddb = beta * self.ddb + (1-beta) * delta_b ** 2
		# self.weights -= delta_w
		# self.biases -= delta_b

		# nesterov: .0005
		# beta = 0.9
		# self.mw = beta * self.mw + learning_rate * (dw - beta * self.mw)
		# self.mb = beta * self.mb + learning_rate * (db - beta * self.mb)
		# self.weights -= self.mw
		# self.biases -= self.mb

		# adam: .0075
		# beta1 = .9
		# beta2 = .999
		# epsilon = 1e-7
		# self.mw = beta1 * self.mw + (1-beta1) * dw
		# self.vw = beta2 * self.vw + (1-beta2) * dw ** 2
		# self.mb = beta1 * self.mb + (1-beta1) * db
		# self.vb = beta2 * self.vb + (1-beta2) * db ** 2
		# mhatw = self.mw / (1-np.power(beta1, self.i))
		# vhatw = self.vw / (1-np.power(beta2, self.i))
		# mhatb = self.mb / (1-np.power(beta1, self.i))
		# vhatb = self.vb / (1-np.power(beta2, self.i))
		# self.i += 1
		# self.weights -= learning_rate * mhatw / np.sqrt(vhatw + epsilon)
		# self.biases -= learning_rate * mhatb / np.sqrt(vhatb + epsilon)

		# adamax: .002
		# beta1 = .9
		# beta2 = .999
		# self.mw = beta1 * self.mw + (1-beta1) * dw
		# self.mb = beta1 * self.mb + (1-beta1) * db
		# mhatw = self.mw / (1 - np.power(beta1, self.i))
		# mhatb = self.mb / (1 - np.power(beta1, self.i))
		# self.vw = np.maximum(self.vw * beta2, np.abs(dw))
		# self.vb = np.maximum(self.vb * beta2, np.abs(db))
		# self.weights -= learning_rate * mhatw / (self.vw + 1e-7)
		# self.biases -= mhatb / (self.vb + 1e-7)
		# self.i += 1

		# nadam: .0075
		# beta1 = .9
		# beta2 = .999
		# epsilon = 1e-7
		# self.vw = beta2 * self.vw + (1-beta2) * dw ** 2
		# self.vb = beta2 * self.vb + (1-beta2) * db ** 2
		# self.mw = beta1 * self.mw + (1-beta1) * dw
		# self.mb = beta1 * self.mb + (1-beta1) * db
		# vhatw = self.vw / (1 - np.power(beta2, self.i))
		# vhatb = self.vb / (1 - np.power(beta2, self.i))
		# mhatw = self.mw / (1 - np.power(beta1, self.i))
		# mhatb = self.mb / (1 - np.power(beta1, self.i))
		# self.i += 1
		# self.weights -= learning_rate / np.sqrt(vhatw + epsilon) * \
		# 				(beta1 * mhatw + (1-beta1) / (1 - np.power(beta1, self.i)) * dw)
		# self.biases -= learning_rate / np.sqrt(vhatb + epsilon) * \
		# 				(beta1 * mhatb + (1 - beta1) / (1 - np.power(beta1, self.i)) * db)

		# AMSGrad: .003
		# beta1 = .9
		# beta2 = .999
		# epsilon = 1e-7
		# self.mw = beta1 * self.mw + (1 - beta1) * dw
		# self.mb = beta1 * self.mb + (1 - beta1) * db
		# self.vw = beta2 * self.vw + (1 - beta2) * dw ** 2
		# self.vb = beta2 * self.vb + (1 - beta2) * db ** 2
		# self.vhatw = np.maximum(self.vhatw, self.vw)
		# self.vhatb = np.maximum(self.vhatb, self.vb)
		# self.weights -= learning_rate * self.mw / np.sqrt(self.vhatw + epsilon)
		# self.biases -= learning_rate * self.mb / np.sqrt(self.vhatb + epsilon)

		# always kept the same
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
		self.data = Data()

	def reset(self):
		for layer in self.layers:
			layer.reset()

	def forward(self, inputs):
		for layer in self.layers[:-1]:
			inputs = self.activation.forward(layer.forward(inputs))
		inputs = self.outputActivation.forward(self.layers[-1].forward(inputs))
		return inputs

	def backward(self, output_gradient, learning_rate):
		for layer in reversed(self.layers):
			output_gradient = layer.backward(output_gradient, learning_rate)
			# L2 Regularization - shape error
			# lamda = .1  # to .01
			# output_gradient += lamda * layer.weights

	def train(self, training_X, training_Y, epochs, learning_rate, verbose_interval):
		for iteration in range(epochs):
			gradient = self.gradient_function.gradient(self.forward(training_X), training_Y)
			self.backward(gradient, learning_rate)
			if verbose_interval > 0 and iteration % verbose_interval == 0:
				print(
					f'\nIteration {iteration} training loss: {np.mean(categorical_cross_entropy(self.forward(self.data.X_train), self.data.Y_train_OHE))}')
				self.print_accuracy()

	def train_stochastic(self, epochs, learning_rate, verbose):
		if verbose: print(f"\nStochastic results with {epochs} epochs,  {learning_rate} alpha: ")
		for e in range(epochs):
			self.data.shuffle()
			for i in range(49999):
				self.train(self.data.X_train[i:i + 1, :], self.data.Y_train_OHE[i:i + 1, :], 1, learning_rate, 0)
			self.calculate_accuracy()
			if verbose:
				print(f'Epoch {e + 1}: ')
				self.print_accuracy()

	def train_minibatch(self, batch_size, epochs, learning_rate, verbose):
		if verbose: print(f"\nMinibatch results with {batch_size} batch size, {learning_rate} alpha: ")
		for e in range(epochs):
			self.data.shuffle()
			for i in range(50000 // batch_size):
				self.train(self.data.X_train[batch_size * i: batch_size * (i + 1), :],
						   self.data.Y_train_OHE[batch_size * i: batch_size * (i + 1), :],
						   1, learning_rate, 0)
			self.calculate_accuracy()
			if verbose:
				print(f'Epoch {e + 1}: ')
				self.print_accuracy()

	def train_largebatch(self, batch_size, iterations):
		alpha = 1 / batch_size
		self.train(self.data.X_train[:batch_size], self.data.Y_train_OHE[:batch_size], iterations, alpha, 50)
		print("\nFinal accuracy: ")
		self.calculate_accuracy()
		self.print_accuracy()

	def calculate_accuracy(self):
		self.predicted = np.argmax(self.forward(self.data.X_train), axis=1)
		self.training_accuracy = (self.predicted == self.data.Y_train).sum() / self.predicted.size
		self.predicted = np.argmax(self.forward(self.data.X_val), axis=1)
		self.validation_accuracy = (self.predicted == self.data.Y_val).sum() / self.predicted.size

	def print_accuracy(self):
		predicted = np.argmax(self.forward(self.data.X_train), axis=1)
		self.training_accuracy = (predicted == self.data.Y_train).sum() / predicted.size
		print("training accuracy: ", self.training_accuracy)

		predicted = np.argmax(self.forward(self.data.X_val), axis=1)
		self.validation_accuracy = (predicted == self.data.Y_val).sum() / predicted.size
		print("validation accuracy: ", self.validation_accuracy)
		print(predicted[:10], "\n", self.data.Y_val[:10])

	def train_stochastic_bootstrap(self, iterations, learning_rate, verbose):
		if verbose: print(f'stochastic bootstrap with {iterations} iterations, {learning_rate} alpha: ')
		for i in range(iterations):
			index = np.random.randint(0, 49999)
			self.train(self.data.X_train[index:index + 1, :], self.data.Y_train_OHE[index:index + 1, :], 1,
					   learning_rate, 0)
			if verbose and i % 10000 == 0:
				print(f'Iteration {i}: ')
				self.print_accuracy()
		self.calculate_accuracy()
		if verbose:
			print('Final accuracy: ')
			self.print_accuracy()

	def train_minibatch_bootstrap(self, batch_size, iterations, learning_rate, verbose):
		if verbose: print(
			f'minibatch bootstrap with {batch_size} batch size, {iterations} iterations, {learning_rate} alpha:')
		for i in range(iterations):
			idx = np.random.randint(0, 49999 // batch_size)
			self.train(self.data.X_train[batch_size * idx: batch_size * (idx + 1), :],
					   self.data.Y_train_OHE[batch_size * idx: batch_size * (idx + 1), :],
					   1, learning_rate, 0)
			if verbose and i % 1000 == 0:
				print(f'Iteration {i}: ')
				self.print_accuracy()
		self.calculate_accuracy()
		if verbose:
			print("Final accuracy: ")
			self.print_accuracy()


class Data:
	def __init__(self):
		self.data = pd.read_csv('mnist/mnist_train.csv')
		self.data = np.array(self.data)
		self.shuffle()

	def shuffle(self):
		np.random.shuffle(self.data)
		self.split()

	def split(self):
		data_train = self.data[:50000]
		data_val = self.data[50000:]

		self.X_train = data_train[:, 1:]
		self.Y_train = data_train[:, 0]
		self.Y_train_OHE = np.zeros((self.Y_train.size, 10))
		self.Y_train_OHE[np.arange(self.Y_train.size), self.Y_train] = 1

		self.X_val = data_val[:, 1:]
		self.Y_val = data_val[:, 0]
		self.Y_val_OHE = np.zeros((self.Y_val.size, 10))
		self.Y_val_OHE[np.arange(self.Y_val.size), self.Y_val] = 1


class Activations:
	def __init__(self):
		self.sigmoid = SigmoidActivation()
		self.softmax = SoftmaxActivation()


class Gradients:
	def __init__(self):
		self.mse_prime = MsePrime()
		self.cce_prime = CcePrime()


class Tester:
	def __init__(self, layer_list):
		activations = Activations()
		gradients = Gradients()
		self.network = Network(layer_list, activations.sigmoid, activations.softmax, gradients.cce_prime)

	def reset(self):
		self.network.reset()

	def stochasticAndMinibatchTest(self, iterations, epochs, learning_rate):
		print(f'Stochastic test with {iterations} iterations of {epochs} epochs, with {learning_rate} alpha: ')
		avg_stochastic_accuracy = 0
		for i in range(iterations):
			self.network.train_stochastic(epochs, learning_rate, False)
			avg_stochastic_accuracy += self.network.validation_accuracy
			self.network.reset()
		print(f"average stochastic accuracy over {iterations} iterations after {epochs} epochs: ",
			  avg_stochastic_accuracy / iterations)

	def minibatchTest(self, iterations, epochs, learning_rate):
		print(f'Minibatch test with {iterations} iterations of {epochs} epochs, with {learning_rate} alpha')
		avg_minibatch_accuracy = 0
		for i in range(iterations):
			self.network.train_minibatch(8, epochs, learning_rate, False)
			avg_minibatch_accuracy += self.network.validation_accuracy
			self.network.reset()
		print(f"average minibatch accuracy over {iterations} iterations after {epochs} epochs: ",
			  avg_minibatch_accuracy / iterations)

	def bootstrapTest(self, iterations, learning_rate):
		print(
			f'Bootstrap stochastic and Minibatch test with {iterations} iterations each, with {learning_rate} alpha: ')
		avg_minibatch_accuracy = 0
		avg_stochastic_accuracy = 0

		for i in range(iterations):
			self.network.train_minibatch_bootstrap(8, 10000, learning_rate, False)
			avg_minibatch_accuracy += self.network.validation_accuracy
			self.network.reset()
			self.network.train_stochastic_bootstrap(100000, learning_rate, False)
			avg_stochastic_accuracy += self.network.validation_accuracy
			self.network.reset()
		print(f"average bootstrap minibatch accuracy after 10000 epochs: ", avg_minibatch_accuracy / iterations)
		print(f"average bootstrap stochastic accuracy after 100000 epochs: ", avg_stochastic_accuracy / iterations)


if __name__ == '__main__':
	tester = Tester([784, 16, 16, 10])
	# tester.network.train_largebatch(50000, 500)
	# tester.network.train_minibatch(8, 10, .003, True)
	# tester.network.train_stochastic(5, .003, True)

	tester.stochasticAndMinibatchTest(5, 5, .003)
