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
img_path = 'D:/BigDataProject/neural_network/face_recognition//image/FaceDB_orl/'


class FaceRecognize(object):
    def __init__(self, num_round=10, learn_rate=0.001, shuffle=False, divide_rate=0.9, units=2019, early_stopping=5, store_model=False):
        self.round = num_round
        self.learn_rate = learn_rate
        self.shuffle = shuffle
        self.divide_rate = divide_rate
        self.units = units
        self.early_stopping = early_stopping
        self.__store_model = store_model
        self.text = ""

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
        self.text = self.text + '\n' + "picture size:\t {}".format(image_value.shape)
        self.text = self.text + '\n' + "label size:\t {}".format(label_value.shape)
        return image_value, label_value

    def divide(self, rate=0.7, shuffle=False):
        """
        Divided the data set to train and test set.
        :param rate:
        :param shuffle:
        :return:
        """
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
        print('Params{\n learning_rate => %.5f\n iteration => %d\n divide_rate => %.2f\n the number of units => %d\n early_stopping => %d\n}\n' %
              (self.learn_rate, self.round, self.divide_rate, self.units, self.early_stopping))
        # image, label = self.load_image()
        train_x, train_y, test_x, test_y = self.divide(rate=self.divide_rate, shuffle=self.shuffle)
        print('train_image_size:', train_x.shape)
        print('train_label_size:', train_y.shape)
        print('test_image_size: ', test_x.shape)
        print('test_label_size: ', test_y.shape)
        self.text = self.text + '\n' + "train_image_size: {}".format(train_x.shape)
        self.text = self.text + '\n' + "train_label_size: {}".format(train_y.shape)
        self.text = self.text + '\n' + "test_image_size:  {}".format(test_x.shape)
        self.text = self.text + '\n' + "test_label_size:  {}".format(test_y.shape)

        input_x = tf.placeholder(tf.float32, [None, 112 * 92], name='input_x')          # None表示张量的第一个维度可以是任意长度
        output_y = tf.placeholder(tf.int32, [None, 40])                                 # 输出：40个数字的标签
        input_x_images = tf.reshape(input_x, [-1, 112, 92, 1])                          # 改变形状之后的输入

        """Construct convolution neural network"""
        """First convolution layer"""
        convolution_one = tf.layers.conv2d(
            inputs=input_x_images,  # 形状是[112, 92, 1]
            filters=32,             # 32个过滤器，输出的深度是32
            kernel_size=[5, 5],     # 过滤器在二维的大小是（5，5）
            strides=1,              # 步长是1
            padding='same',         # same表示输出的大小不变，因此需要在外围补0两圈
            activation=tf.nn.relu   # 激活函数是Relu
        )                           # 形状[112, 92, 32]

        """First pooling layer"""
        pool_one = tf.layers.max_pooling2d(
            inputs=convolution_one,      # 形状[112, 92, 32]
            pool_size=[2, 2],            # 过滤器在二维的大小是（2 * 2）
            strides=2                    # 步长是2
        )                                # 形状[56, 46, 32]

        """Second convolution layer"""
        convolution_two = tf.layers.conv2d(
            inputs=pool_one,          # 形状是[56, 46, 32]
            filters=64,               # 64个过滤器，输出的深度是64
            kernel_size=[5, 5],       # 过滤器在二维的大小是（5，5）
            strides=1,                # 步长是1
            padding='same',           # same表示输出的大小不变，因此需要在外围补0两圈
            activation=tf.nn.relu     # 激活函数是Relu
        )                             # 形状[56, 46, 64]

        """Second pooling layer"""
        pool_two = tf.layers.max_pooling2d(
            inputs=convolution_two,        # 形状[56, 46, 64]
            pool_size=[2, 2],              # 过滤器在二维的大小是（2 * 2）
            strides=2                      # 步长是2
        )                                  # 形状[28, 23, 64]

        """flatting"""
        flat = tf.reshape(pool_two, [-1, 28 * 23 * 64])          # 形状[7 * 7 * 64, ]

        """The number of neural cell in fully-connected layer"""
        dense = tf.layers.dense(inputs=flat, units=self.units, activation=tf.nn.relu)

        # Dropout:50%，rate=0.5
        dropout = tf.layers.dropout(inputs=dense, rate=0.5, name='dropout')

        """Output layer, there needn't activation function"""
        logit = tf.layers.dense(inputs=dropout, units=40, name='logit')     # 输出。形状[1, 1, 10]

        """Calculation error - Cross entropy then compute percent probability with Softmax"""
        loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y, logits=logit)

        # use Adam Optimizer to minimize error with learning_rate
        train_op = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(loss)    # learn_rate

        """Compute accuracy of test data set"""
        # accuracy = tf.metrics.accuracy(labels=tf.argmax(output_y), predictions=tf.argmax(logit))[1]
        correct_prediction = tf.equal(tf.argmax(output_y), tf.argmax(logit))    # return a string of bool value
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        epoch = 0     # for counting

        with tf.Session() as sess:
            """Initial global and local variables"""
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)
            print('\n-------------start to train model------------')
            self.text = self.text + '\n' + "\n------------start to train model------------"
            count, formal_score = 0, 0
            for i in range(self.round):
                """start graph computation"""
                epoch = i
                train_loss, train_op_ = sess.run([loss, train_op], feed_dict={input_x: train_x, output_y: train_y})
                if i % 1 == 0:
                    test_accuracy = sess.run(accuracy, feed_dict={input_x: test_x, output_y: test_y})
                    print("Step=%d, Train loss=%.3f, [Test accuracy=%.3f]" % (i+1, train_loss, test_accuracy))
                    self.text = self.text + '\n' + "Step={}, Loss={:.4f}, [Accuracy={:.3f}]".format(i+1, train_loss, test_accuracy)

                    """Test a sample output 
                    test = self.load_predict_sample('image/10.png')
                    output = sess.run(logit, feed_dict={input_x: test})
                    print('prediction: ', output)
                    """
                    # print(logit)

                    if test_accuracy <= formal_score:
                        count += 1
                        if count >= self.early_stopping:
                            print('early stopping.')
                            self.text = self.text + '\n' + "early stopping."
                            # if self.__store_model is True:
                            #     self.store_model(sess, epoch)
                            break
                        else:
                            continue
                    else:
                        formal_score = test_accuracy
                        count = 0

            if self.__store_model is True:
                self.store_model(sess, epoch)

        self.text = self.text + '\n' + "over."
        return self.text

    def store_model(self, sess, epoch):
        absolute_path = 'D:/BigDataProject/neural_network/face_recognition/'
        saver = tf.train.Saver()     # Save model, maximun to save: 5
        my_file = absolute_path + 'model'
        if os.path.exists(my_file) is False:
            os.mkdir(my_file)
            print('make directory successful.')
            self.text = self.text + '\n' + "make directory successful."
        else:
            file_name = os.listdir(my_file)
            if len(file_name) != 0:
                for string in file_name:
                    os.remove(my_file + '/' + string)   # delete files

        saver.save(sess, my_file + "/cnn_multi_classifier", global_step=epoch)     # global_step
        print('model store successfully.')
        self.text = self.text + '\n' + "model store successfully."

    def load_model(self, test):
        absolute_path = 'D:/BigDataProject/neural_network/face_recognition/'
        if os.path.exists(absolute_path + 'model') is True:
            dir_list = os.listdir(absolute_path + 'model')
            if len(dir_list) == 0:
                print('model is empty.')
                exit(0)
            else:
                with tf.Session() as sess:
                    path = absolute_path + 'model/cnn_multi_classifier-19.meta'
                    for file in dir_list:
                        if 'meta' in file:
                            path = absolute_path + 'model/' + file
                    saver = tf.train.import_meta_graph(path)
                    saver.restore(sess, tf.train.latest_checkpoint(absolute_path + "model/"))
                    print('load model successfully.')

                    graph = tf.get_default_graph()
                    input_x = graph.get_tensor_by_name('input_x:0')
                    logit = graph.get_tensor_by_name('logit/BiasAdd:0')

                    feed_dict = {input_x: test}
                    result = sess.run(logit, feed_dict=feed_dict)
                    max_index = sess.run(tf.argmax(result, axis=1))     # find the indices of maximum value
                    print('Prediction result:', result)
                    print('This image is %d th people.' % (max_index[0] + 1))
                    return max_index[0] + 1
        else:
            print('directory is not exist.')
            exit(0)

    def load_predict_sample(self, filename):
        file_name = filename
        image = cv2.imread(file_name)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).reshape(1, -1)
        # image = np.array(list(image[0]))
        # print(image)
        return image


if __name__ == '__main__':
    face = FaceRecognize(num_round=10, learn_rate=0.0005, shuffle=False, divide_rate=0.9, units=2019, early_stopping=10, store_model=False)
    print("Start to running.")
    # face.run()
    test_sample = face.load_predict_sample('image/10.png')
    class_name = face.load_model(test_sample)
    print(class_name)


"""
Two blogs:
first:store and restore tensorflow model
two:the whole project.
"""
