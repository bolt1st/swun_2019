import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image


datagen = ImageDataGenerator(
        rotation_range=60,            # 随机旋转的度数范围
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1.0/255,               # 重缩放因子(数据乘以所提供的值)
        shear_range=0.1,              # 剪切强度（以弧度逆时针方向剪切角度）
        zoom_range=0.1,               # 随机缩放范围
        horizontal_flip=True,         # 随机水平翻转
        fill_mode='nearest'           # {"constant", "nearest", "reflect" or "wrap"} 之一。默认为 'nearest'。输入边界以外的点根据给定的模式填充
)

train_corn_dir = 'C:\\Users\\wangjipeng.IMAGEDESIGN\\Desktop\\开题相关\\data\\玉米大斑病'
fname = [os.path.join(train_corn_dir, fname) for fname in os.listdir(train_corn_dir)]
# 选择一个图像

for j in range(5):
        img_path = fname[j]
# 读取图片并调整大小为150*150
# img = Image.open(img_path)
# img1 = img.Image.resize(150, 150)
        img = image.load_img(img_path, target_size=(150, 150))
# 把图片转换成shape为（150*150）的张量
        x = image.img_to_array(img)
# reshape，使得(1, 150, 150, 3)
        x = x.reshape((1,) + x.shape)

# 下面是产生图片的代码，产生的图片保存在‘train’目录下
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir='C:\\Users\\wangjipeng.IMAGEDESIGN\\Desktop\\开题相关\\data\\train\\leaf_blight', save_prefix='leaf_blight', save_format='jpg'):
                i += 1
                if i > 49:
                        break