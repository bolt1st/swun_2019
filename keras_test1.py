from keras.models import Sequential
from keras.layers.convolutional import Conv2D                 # 卷积层，图像的空间卷积
from keras.layers.convolutional import MaxPooling2D           # 池化层
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K


class LeNet:
    @