import pandas as pd
import numpy as np

# np.random.seed(0)
data = pd.read_csv('mnist/mnist_train.csv')
data = np.array(data)

np.random.shuffle(data)
data_train = data[:50000]
data_val = data[50000:]

X_train = data_train[:, 1:]
Y_train = data_train[:, 0]
Y_train_OHE = np.zeros((Y_train.size, 10))
Y_train_OHE[np.arange(Y_train.size), Y_train] = 1

X_val = data_val[:, 1:]
Y_val = data_val[:, 0]
Y_val_OHE = np.zeros((Y_val.size, 10))
Y_val_OHE[np.arange(Y_val.size), Y_val] = 1

