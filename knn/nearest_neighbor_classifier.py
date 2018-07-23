import numpy as np
import operator


class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        """ X is N x D ,每一行代表一条测试数据，y是N维向量，代表每条测试数据的标签 """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=2):
        """X 是需要预测的数据 N x D，每一行代表一条待预测数据"""

        if num_loops == 1:
            dists = self.computer_distance_one_loop(X)
        elif num_loops == 2:
            dists = self.computer_distance_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k)

    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        dists = dists.argsort()
        for i in range(num_test):
            curr_label = self.y_train[dists[i, 0:k]]
            class_count = {}
            for i in range(k):
                near_label = curr_label[i]
                class_count[near_label] = class_count.get(near_label, 0) + 1
            label = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)[0][0]
            y_pred[i] = label
            print(label)
        return y_pred

    def computer_distance_two_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        for i in range(num_test):
            for j in range(num_train):
                dists[i, j] = np.sqrt(np.sum((X[i, :] - self.X_train[j, :]) ** 2))
        return dists

    def computer_distance_one_loop(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_train):
            dists[i, :] = np.sqrt(np.sum(np.square(self.X_train - X[i, :]), axis=(1, 2)))
        return dists
