# -*- coding:utf-8 -*-

import os
import sys
import cv2
import random
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
img_path = './image/FaceDB_orl/'


class FaceRecognize(object):
    def __init__(self, num_round=10, learn_rate=0.001, shuffle=False, divide_rate=0.9, units=1024):
        self.round = num_round
        self.learn_rate = learn_rate
        self.shuffle = shuffle
        self.divide_rate = divide_rate
        self.units = units
        print('Params{\nlearning_rate => %f\n iteration => %f\n divide_rate => %f\n the number of units => %f\n}' % (self.round, self.round, self.divide_rate, self.units))

    def load_image(self):
        image_value = []
        label_value = np.array([[0] * 40 for _ in range(400)])

        directory = os.listdir(img_path)
        for index in range(len(directory)):
            sub_dir = img_path + directory[index]
            file_names = os.listdir(sub_dir)
            for i in range(len(file_names)):
                name = sub_dir + '/' + file_names[i]
                image = cv2.imread(name)                                            # 考虑是否使用RGB格式，极大增加计算量
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).reshape(1, -1)      # 使用灰度图的值
                # image_value = np.vstack((image_value, image))
                image_value.append(list(image[0]))
                # print(index, i)   # print the indices
                label_value[index * 10 + i][index] = 1

        image_value = np.array(image_value)
        print('picture size:\t', image_value.shape)                    # (400, 10304)
        print('label size:\t\t', label_value.shape)                    # (400, 40)
        return image_value, label_value

    def divide(self, rate=0.7, shuffle=False):
        image, label = self.load_image()
        train, test = [], []
        if shuffle is False:
            for index in range(400):
                if index % 10 >= (rate * 10):
                    test.append(index)
                else:
                    train.append(index)
        else:
            indices = [index for index in range(400)]
            index = int(rate * 399)
            random.shuffle(indices)
            train = indices[:index]
            test = indices[index:]

        # print(image[train].shape, label[train].shape)
        # print(image[test].shape, label[test].shape)
        return image[train], label[train], image[test], label[test]

    def run(self):
        # image, label = self.load_image()
        train_x, train_y, test_x, test_y = self.divide(rate=self.divide_rate, shuffle=self.shuffle)
        print('train_image_size:', train_x.shape)
        print('train_label_size:', train_y.shape)
        print('test_image_size: ', test_x.shape)
        print('test_label_size: ', test_y.shape)

        # None表示张量的第一个维度可以是任意长度
        input_x = tf.placeholder(tf.float32, [None, 112 * 92])          # 灰度值：0～255
        output_y = tf.placeholder(tf.int32, [None, 40])                 # 输出：10个数字的标签
        input_x_images = tf.reshape(input_x, [-1, 112, 92, 1])          # 改变形状之后的输入

        # 构建卷积神经网络
        # 第 1 层卷积
        conv1 = tf.layers.conv2d(inputs=input_x_images,  # 形状是[112, 92, 1]
                                 filters=32,             # 32个过滤器，输出的深度是32
                                 kernel_size=[5, 5],     # 过滤器在二维的大小是（5，5）
                                 strides=1,              # 步长是1
                                 padding='same',         # same表示输出的大小不变，因此需要在外围补0两圈
                                 activation=tf.nn.relu   # 激活函数是Relu
                                 )                       # 形状[112, 92, 32]

        # 第 1 层池化(亚采样)
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1,      # 形状[112, 92, 32]
            pool_size=[2, 2],  # 过滤器在二维的大小是（2 * 2）
            strides=2          # 步长是2
        )                      # 形状[56, 46, 32]

        # 第 2 层卷积
        conv2 = tf.layers.conv2d(inputs=pool1,          # 形状是[56, 46, 32]
                                 filters=64,            # 64个过滤器，输出的深度是64
                                 kernel_size=[5, 5],    # 过滤器在二维的大小是（5，5）
                                 strides=1,             # 步长是1
                                 padding='same',        # same表示输出的大小不变，因此需要在外围补0两圈
                                 activation=tf.nn.relu  # 激活函数是Relu
                                 )                      # 形状[56, 46, 64]

        # 第 2 层池化(亚采样)
        pool2 = tf.layers.max_pooling2d(
            inputs=conv2,        # 形状[56, 46, 64]
            pool_size=[2, 2],    # 过滤器在二维的大小是（2 * 2）
            strides=2            # 步长是2
        )                        # 形状[28, 23, 64]

        # 平坦化（flat）
        flat = tf.reshape(pool2, [-1, 28 * 23 * 64])  # 形状[7 * 7 * 64, ]

        # 1024个神经元的全连接层
        dense = tf.layers.dense(inputs=flat, units=self.units, activation=tf.nn.relu)

        # Dropout:丢弃50%，rate=0.5
        dropout = tf.layers.dropout(inputs=dense, rate=0.5)

        # 40个神经元的全连接层，这里不用激活函数来做非线性激活了
        logit = tf.layers.dense(inputs=dropout, units=40)  # 输出。形状[1, 1, 10]

        # 计算误差（计算Cross entropy（交叉熵），再用Softmax计算百分比概率）
        loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y, logits=logit)

        # Adam 优化器来最小化误差，学习率0.01
        train_op = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(loss)    # learn_rate

        # 精度计算 预测值 和 实际标签的匹配程度
        # 返回（accuracy, update_op）,会创建两个局部变量
        # accuracy = tf.metrics.accuracy(labels=tf.argmax(output_y, axis=1), predictions=tf.argmax(logit, axis=1))[1]
        # accuracy = tf.metrics.accuracy(labels=tf.argmax(output_y), predictions=tf.argmax(logit))[1]

        correct_prediction = tf.equal(tf.argmax(output_y), tf.argmax(logit))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.Session() as sess:
            # 初始化全局和局部变量
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)
            print('-------------start to train model------------')
            for i in range(self.round):
                # batch = mnist.train.next_batch(50)  # 从训练集中取下一个50个样本
                train_loss, train_op_ = sess.run([loss, train_op], feed_dict={input_x: train_x, output_y: train_y})
                if i % 1 == 0:
                    test_accuracy = sess.run(accuracy, feed_dict={input_x: test_x, output_y: test_y})
                    print("Step=%d, Train loss=%.4f, [Test accuracy=%.4f]" % (i+1, train_loss, test_accuracy))


if __name__ == '__main__':
    face = FaceRecognize(num_round=50, learn_rate=0.0001, shuffle=False, divide_rate=0.9, units=2019)
    face.run()

