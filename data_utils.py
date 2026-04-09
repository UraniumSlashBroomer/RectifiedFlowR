import pickle
import numpy as np
import os
from torch.utils.data import DataLoader

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

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, subtract_mean=True):
    cifar10_dir = os.path.join(
            os.path.dirname(__file__), "dataset/cifar-10-batches-py"
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

    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    result_dict = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test
    }

    return result_dict

if __name__ == '__main__':
    result_data = get_CIFAR10_data()
    batch_size = 16
    dataloader = DataLoader(result_data["X_train"], batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"CIFAR loaded correctly, num of samples when batch_size={batch_size}: {len(dataloader) * 16}")
