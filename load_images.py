# -*- coding:utf-8 -*-

"""Face database
https://www.cnblogs.com/kuangqiu/p/7776829.html
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
img_path = './image/FaceDB_orl/'


def load_image():
    image_value = []
    label_value = np.array([[0] * 40 for _ in range(400)])

    directory = os.listdir(img_path)
    for index in range(len(directory)):
        sub_dir = img_path + directory[index]
        file_names = os.listdir(sub_dir)
        for i in range(len(file_names)):
            name = sub_dir + '/' + file_names[i]
            image = cv2.imread(name)         # 考虑是否使用RGB格式，极大增加计算量
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).reshape(1, -1)      # 使用灰度图的值
            # image_value = np.vstack((image_value, image))
            image_value.append(list(image[0]))
            # print(index, i)   # print the indices
            label_value[index * 10 + i][index] = 1

    image_value = np.array(image_value)
    print('picture size:\t', image_value.shape)                    # (400, 10304)
    print('label size:\t\t', label_value.shape)                    # (400, 40)
    return image_value, label_value


if __name__ == '__main__':
    load_image()

