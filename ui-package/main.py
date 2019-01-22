# -*- coding: utf-8 -*-

import gc
import cv2
import sys
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import QtGui, QtWidgets
from MainInterface import Ui_MainWindow
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QGraphicsScene, QGraphicsPixmapItem
sys.path.append('E:/comtemporary/face_recognition/')
from face_recognize import FaceRecognize


class MyWindow(QMainWindow, Ui_MainWindow):     # 这个窗口继承了用 Qt Designner 绘制的窗口

    def __init__(self):
        super(MyWindow, self).__init__()        # super(MyWindow,self) 首先找到 MyWindow 的父类（就是类 Ui_MainWindow），然后把类 MyWindow 的对象 MyWindow 转换为类 Ui_MainWindow 的对象
        self.setupUi(self)

    def start_button(self):
        # self.lineEdit_4.setText("7")    # set context   attention: use double quotation marks (" ")
        # self.checkBox.checkState()      # return state of checkBox
        self.textBrowser.setText("")

        temp = self.checkBox.checkState()
        store_model = True if temp == 2 else False

        temp = self.checkBox_2.checkState()
        shuffle = True if temp == 2 else False

        learning_rate = float(self.lineEdit.text())
        iteration = int(self.lineEdit_2.text())
        units = int(self.lineEdit_3.text())
        early_stopping = int(self.lineEdit_4.text())
        divide_rate = float(self.lineEdit_6.text())

        # print(store_model)      # <class 'bool'>
        # print(shuffle)          # <class 'bool'>
        # print(learning_rate)    # <class 'float'>
        # print(iteration)        # <class 'int'>
        # print(units)            # <class 'int'>
        # print(early_stopping)   # <class 'int'>
        # print(divide_rate)      # <class 'float'>
        # # exit(0)

        """Create face object"""
        face = FaceRecognize(num_round=iteration, learn_rate=learning_rate, shuffle=shuffle, divide_rate=divide_rate, units=units, early_stopping=early_stopping, store_model=store_model)
        show_text = "Params(\n store_model => {} \n shuffle => {} \n learning_rate => {} \n iteration => {} \n divide_rate => {} \n the number of units => {} \n early_stopping => {}\n)".format(store_model, shuffle, learning_rate, iteration, divide_rate, units, early_stopping)
        return_text = face.run()
        show_text = show_text + return_text

        # self.textBrowser.setText("Params(\n store_model => {} \n shuffle => {} \n learning_rate => {} \n iteration => {} \n divide_rate => {} \n the number of units => {} \n early_stopping => {}\n)".format(store_model, shuffle, learning_rate, iteration, divide_rate, units, early_stopping))
        self.textBrowser.setText(show_text)

        """Destroy object:face"""
        del face
        gc.collect()

    def tool_button_click(self):
        print('Tool button clicked.')
        self.graphicsView.setEnabled(True)
        self.graphicsView_2.setEnabled(True)

        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "E:/comtemporary/face_recognition/image/FaceDB_orl/", " *.png;;*.jpg;;*.jpeg;;*.bmp;;All Files (*)")    # " *.jpg;;*.png;;*.jpeg;;*.bmp")
        print(str(imgName))
        self.lineEdit_5.setText(imgName.split('/')[-1])      # 显示文件名

        """show selected picture"""
        img = cv2.imread(imgName)                          # 读取图像
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)         # 转换图像通道
        x = img.shape[1]                                   # 获取图像大小
        y = img.shape[0]
        frame = QImage(img, x, y, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        item = QGraphicsPixmapItem(pix)                    # 创建像素图元
        scene = QGraphicsScene()                           # 创建场景
        scene.addItem(item)
        self.graphicsView.setScene(scene)                  # 将场景添加至视图

        face = FaceRecognize()
        test_sample = face.load_predict_sample(imgName)
        class_name = face.load_model(test_sample)

        """show predicted picture"""
        if class_name < 10:
            image_path = 'E:/comtemporary/face_recognition/image/FaceDB_orl/00'
        else:
            image_path = 'E:/comtemporary/face_recognition/image/FaceDB_orl/0'

        file_name = image_path + str(class_name) + '/01.png'
        print(file_name)

        img_2 = cv2.imread(file_name)                           # 读取图像
        # img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)        # 转换图像通道
        x_2 = img.shape[1]                                      # 获取图像大小
        y_2 = img.shape[0]
        frame_2 = QImage(img_2, x_2, y_2, QImage.Format_RGB888)
        pix_2 = QPixmap.fromImage(frame_2)
        item_2 = QGraphicsPixmapItem(pix_2)                     # 创建像素图元
        scene_2 = QGraphicsScene()                              # 创建场景
        scene_2.addItem(item_2)
        self.graphicsView_2.setScene(scene_2)                   # 将场景添加至视图

        print(int(imgName.split('/')[-2]))
        print(class_name)

        if int(imgName.split('/')[-2]) == class_name:
            self.textBrowser_2.setText("Congratulations, classify correctly.\nThis picture is the {} th people.".format(class_name))
        else:
            self.textBrowser_2.setText("I'm so sorry, classify mistakenly.\n\nThis picture is the {} th people. \nNot {} th people.".format(int(imgName.split('/')[-2]), class_name))

        """Destroy object:face"""
        del face
        gc.collect()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MyWindow()
    MainWindow.show()
    sys.exit(app.exec_())


"""自定义功能区
self.pushButton.clicked.connect(MainWindow.start_button)
self.graphicsView.setEnabled(False)
self.graphicsView_2.setEnabled(False)
self.toolButton.clicked.connect(MainWindow.tool_button_click)
"""

# Process finished with exit code -1073740791 (0xC0000409)
