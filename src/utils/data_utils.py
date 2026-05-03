import pickle
import numpy as np
import os
import torch


def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict["data"]
        Y = datadict["labels"]
        X = X.reshape(10000, 3, 32, 32).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(root):
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(root, "data_batch_%d" % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)

    Xte, Yte = load_CIFAR_batch(os.path.join(root, "test_batch"))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, dtype=torch.float32):
    cifar10_dir = os.path.join(
            os.path.dirname(__file__), "../dataset/cifar-10-batches-py"
    )
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    mask = [i for i in range(num_training, num_training + num_validation)]
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = [i for i in range(num_training)]
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = [i for i in range(num_test)]
    X_test = X_test[mask]
    y_test = y_test[mask]
    
    X_train, X_val, X_test = X_train / 255, X_val / 255, X_test / 255 # pixels are in [0, 1] range
    
    X_train = 2 * X_train - 1 # [-1, 1]
    X_val = 2 * X_val - 1
    X_test = 2 * X_test - 1

    result_dict = {
        "X_train": torch.tensor(X_train, dtype=dtype),
        "y_train": torch.tensor(y_train, dtype=dtype),
        "X_val": torch.tensor(X_val, dtype=dtype),
        "y_val": torch.tensor(y_val, dtype=dtype),
        "X_test": torch.tensor(X_test, dtype=dtype),
        "y_test": torch.tensor(y_test, dtype=dtype)
    }

    return result_dict
