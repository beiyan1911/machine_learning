import pickle
import numpy as np
import os
from cs231n.knn.nearest_neighbor_classifier import NearestNeighbor


def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        x = datadict[b'data']
        y = datadict[b'labels']

        x = x.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
        y = np.array(y)
        return x, y


def load_cifar10(root):
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(root, 'data_batch_%d' % (b,))
        x, y = load_cifar_batch(f)
        xs.append(x)
        ys.append(y)

    Xtrain = np.concatenate(xs)
    Ytrain = np.concatenate(ys)
    del x, y
    Xtest, Ytest = load_cifar_batch(os.path.join(root, 'test_batch'))
    return Xtrain, Ytrain, Xtest, Ytest


def test():
    x_train, y_train, x_test, y_test = load_cifar10('../data/cifar-10-batches-py')

    # x_train = x_train[0:100, :]
    # y_train = y_train[0:100]
    # x_test = x_test[0:10, :]
    # y_test = y_test[0:10]

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    classifier = NearestNeighbor()
    classifier.train(x_train, y_train)
    result = classifier.predict(x_test, k=3)
    print(result)


test()
