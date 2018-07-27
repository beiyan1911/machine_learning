import gzip
import os
from struct import unpack


class MnistData:

    def __init__(self, batch=10, dataset='train', path='.'):
        """
        初始化
        :param batch: 每次读取批量大小
        :param dataset: 数据集的种类
        :param path: 路径
        """
        self.batch = batch  # 计数
        self.num = 0  # 批次数

        if dataset == 'train':
            self.fImages = gzip.open(os.path.join(path, 'train-images-idx3-ubyte.gz'), 'rb')
            self.fLabels = gzip.open(os.path.join(path, 'train-labels-idx1-ubyte.gz'), 'rb')
        elif dataset == 'test':
            self.fImages = gzip.open(os.path.join(path, 't10k-images-idx3-ubyte.gz'), 'rb')
            self.fLabels = gzip.open(os.path.join(path, 't10k-images-idx3-ubyte.gz'), 'rb')
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

    def next_batch(self, only01=False):
        """
        获取下一批次数据
        :param only01: true：x false：【0，1，2，3...x,...9】
        :return:
        """

        # seek to the image we want to start on
        self.fImages.seek(16 + self.num * self.batch + self.imageRows * self.imageCols)
        self.fLabels.seek(8 + self.num * self.batch)
        count = self.batch if (self.num + 1) * self.batch <= self.imageNum else self.imageNum - self.num * self.batch

        images = []
        labels = []

        for blah in range(0, count):
            # get the input from the image file
            x = []
            for i in range(0, self.imageRows * self.imageCols):
                val = 1 if unpack('>B', self.fImages.read(1))[0] > 0 else 0
                x.append(val)
            images.append(x)

            # get the correct label
            val = unpack('>B', self.fLabels.read(1))[0]
            if only01:
                labels.append(val)
            else:
                temp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                temp[val] = 1
                labels.append(temp)
        self.num = self.num + 1
        return images, labels

    def __del__(self):
        # 关闭文件流
        self.fImages.close()
        self.fLabels.close()
