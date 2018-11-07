from keras.models import Sequential
from keras.layers.convolutional import Conv2D                 # 卷积层，图像的空间卷积
from keras.layers.convolutional import MaxPooling2D           # 池化层
from keras.layers.core import Activation                      # 将激活函数应用于输出
from keras.layers.core import Flatten                         # 将输入展平，用于卷积层到全连接层的过渡
from keras.layers.core import Dense                           # 普通的全连接层
from keras import backend as K                                # 导入后端模块


class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)
        # if we are using "channels last", update the input shape
        if K.image_data_format() == "channels_first":               # for Tensorflow
            inputShape = (depth, height, width)
        # first set of CONV==>RELU==>POOL layers
        model.add(Conv2D(20, (5, 5), padding="same", input_sharp=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # second set of CONV==>RELU==>POOL layers
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))


        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))


        # return the constructed network architecture
        return model
        