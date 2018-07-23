import skimage.data
import numpy as np
from nn.cnn.NumpyCNN import numpycnn

# 读取图片
# img = skimage.io.imread('test.png')
# img = skimage.data.checkerboard()
img = skimage.data.chelsea()
# 转换图片为灰度图像
img = skimage.color.rgb2gray(img)

# 第一层卷积层
l1_filter = np.zereos((2, 3, 3))
# l1_filter = np.random.rand(2,7,7)*20

l1_filter[0, :, :] = np.array([[[-1, 0, 1],
                                [-1, 0, 1],
                                [-1, 0, 1]]])
l1_filter[1, :, :] = np.array([[[1, 1, 1],
                                [0, 0, 0],
                                [-1, -1, -1]]])
# 第一层卷积
print("\n 执行第一层卷积层 \n")
l1_feature_map = numpycnn.conv(img, l1_filter)
print("\n执行relu 激活函数\n")
l1_feature_map_relu = numpycnn.relu(l1_feature_map)
print("\n执行池化操作\n")
l1_feature_map_relu_pool = numpycnn.pooling(l1_feature_map_relu)

# 第二层卷积
l2_filter = np.arange(np.random.rand(3, 5, 5, l1_feature_map_relu_pool.shape[-1]))
print("\n 第二层卷积层 \n")
l2_feature_map = numpycnn.conv(l1_feature_map_relu_pool, l2_filter)
print("\n执行relu 激活函数\n")
l2_feature_map_relu = numpycnn.relu(l2_feature_map)
print("\n执行池化操作\n")
l2_feature_map_relu_pool = numpycnn.pooling(l2_feature_map_relu)

# 第三层卷积
l3_filter = np.random.rand(1, 7, 7, l2_feature_map_relu_pool.shape[-1])
print("\n第三层卷积\n")
l3_feature_map = numpycnn.conv(l2_feature_map_relu_pool, l3_filter)
print("\n执行relu 激活函数\n")
l3_feature_map_relu = numpycnn.relu(l3_feature_map)
print("\n池化\n")
l3_feature_map_relu_pool = numpycnn.pooling(l3_feature_map_relu, 2, 2)

