import gzip
import os
from struct import unpack
import numpy as np


class MnistData:

    def __init__(self, batch=10, dataset='train', path='.'):
        """
        初始化
        :param batch: 每次读取批量大小
        :param dataset: 数据集的种类
        :param path: 路径
        """
        self.batch = batch  # 计数
        self.curNum = 0  # 批次数
        self.numChannels = 1
        self.numLabels = 10

        if dataset == 'train':
            self.fImages = gzip.open(os.path.join(path, 'train-images-idx3-ubyte.gz'), 'rb')
            self.fLabels = gzip.open(os.path.join(path, 'train-labels-idx1-ubyte.gz'), 'rb')
        elif dataset == 'test':
            self.fImages = gzip.open(os.path.join(path, 't10k-images-idx3-ubyte.gz'), 'rb')
            self.fLabels = gzip.open(os.path.join(path, 't10k-labels-idx1-ubyte.gz'), 'rb')
        else:
            raise ValueError("dataset must be 'testing' or 'training'")

        # read the header information in the images file.
        s1, s2, s3, s4 = self.fImages.read(4), self.fImages.read(4), self.fImages.read(4), self.fImages.read(4)
        self.imageMagic = unpack('>I', s1)[0]  # magic number
        self.imageNum = unpack('>I', s2)[0]  # image number
        self.imageRows = unpack('>I', s3)[0]
        self.imageCols = unpack('>I', s4)[0]
        # read labels info
        self.labelMagic = unpack('>I', self.fLabels.read(4))[0]
        self.labelNum = unpack('>I', self.fLabels.read(4))[0]
        self.totalNum = self.imageNum / self.batch

    def next_batch(self):
        """
        获取下一批次数据
        :param only01: true：x false：【0，1，2，3...x,...9】
        :return:
        """

        # seek to the image we want to start on
        self.fImages.seek(16 + self.curNum * self.batch * self.imageRows * self.imageCols)
        self.fLabels.seek(8 + self.curNum * self.batch)

        if (self.curNum + 1) * self.batch <= self.imageNum:
            count = self.batch
        else:
            count = self.imageNum - self.curNum * self.batch

        images = np.zeros((self.batch, self.imageRows * self.imageCols), dtype=np.float32)
        labels = np.zeros(self.batch, dtype=np.int16)

        for index in range(count):
            for i in range(self.imageRows * self.imageCols):
                # images[index][i] = 1 if unpack('>B', self.fImages.read(1))[0] > 0 else 0
                images[index][i] = unpack('>B', self.fImages.read(1))[0] / 255
            labels[index] = unpack('>B', self.fLabels.read(1))[0]

        self.curNum = self.curNum + 1

        return images, labels

    def get_all_data(self):
        """
        读取所有数据
        """
        train_data = np.zeros((self.imageNum, self.imageRows * self.imageCols), dtype=np.float32)
        train_label = np.zeros((self.imageNum,), dtype=np.int16)

        for i in range(int(self.totalNum)):
            batch_images, batch_labels = self.next_batch()
            for j in range(self.batch):
                train_data[i * self.batch + j] = batch_images[j]
                train_label[i * self.batch + j] = batch_labels[j]
        return self.__reformat__(train_data, train_label)

    def __reformat__(self, dataset, labels):
        dataset = dataset.reshape((-1, self.imageRows, self.imageCols, self.numChannels)).astype(np.float32)
        labels = (np.arange(self.numLabels) == labels[:, None]).astype(np.float32)
        return dataset, labels

    def __del__(self):
        # 关闭文件流
        self.fImages.close()
        self.fLabels.close()


def get_train_data():
    localPath = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(localPath, '../../../data/mnist')
    X_train, Y_train = MnistData(path=path, dataset='train').get_all_data()
    return X_train, Y_train


def get_test_data():
    localPath = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(localPath, '../../../data/mnist')
    submission_dataset, submission_label = MnistData(path=path, dataset='test').get_all_data()
    return submission_dataset, submission_label
