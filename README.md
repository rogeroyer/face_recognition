<h2 align="center">人脸识别小项目</h2>

### 系统功能
要求输入一张人脸图片后识别出是具体某个人，验证集评价指标采用准确率。 

***
### 人脸库简介
- ORL人脸数据库

共有40个不同年龄、不同性别和不同种族的对象，每个对象10副灰度图像，共计400副灰度图像，图像尺寸是92*112像素。人脸部分表情有变化，如笑与不笑、眼睛睁与不睁、眼镜戴与不戴等，是目前使用最为广泛的标准数据库。

- Yale 人脸数据库

共有15个人，每人11副，共计165副在不同光照、不同表情的人脸图像。

- 由于两个人脸库的图片大小以及格式不一致，所以我只选择了第一个人脸库的人脸进行识别。人脸库见目录`image`。

***
### 核心技术
卷积神经网络(Convolutional Neural Networks, CNN), 是一种前馈神经网络，它的人工神经元可以响应一部分覆盖范围内的周围单元，对于大型图像处理有出色表现。卷积神经网络由一个或多个卷积层和顶端的全连通层（对应经典的神经网络）组成，同时也包括关联权重和池化层（pooling layer）。这一结构使得卷积神经网络能够利用输入数据的二维结构。与其他深度学习结构相比，卷积神经网络在图像和语音识别方面能够给出更好的结果。这一模型也可以使用反向传播算法进行训练。相比较其他深度、前馈神经网络，卷积神经网络需要考量的参数更少，使之成为一种颇具吸引力的深度学习结构。

***
### 运行环境
- Hardware：PC
- OS：Window10
- Software or Package：
  - Python3.6.2
  - tensorflow-1.8.0
  - Pycharm-2018
  - PyQt5
  - opencv-python-3.4.0(cv2)
  - numpy-1.15.4

### 卷积神经网络算法流程
```
第一步：使用5*5大小的卷积核进行卷积，步长设为1，设置ReLu(线性整流)激活函数
第二步：设置2*2大小的池化层，步长设为2
第三步：使用5*5大小的卷积核进行卷积，步长设为1，设置ReLu(线性整流)激活函数
第四步：设置2*2大小的池化层，步长设为2
第五步：设置含有2019个神经元的全连接层，设置ReLu(线性整流)激活函数
第六步：随机丢弃50%的神经元输出结果，防止过拟合
第七步：设置含有40个神经元的输出层，每一个神经元输出对应每一个类别的预测概率
第八步：使用softmax交叉熵作为损失函数以修正模型
```

### 界面展示
![程序主界面](https://github.com/rogeroyer/face_recognition/blob/master/image/MainWindow.png)

***
### 界面功能展示
![功能展示](https://github.com/rogeroyer/face_recognition/blob/master/image/FunctionIntroduction.png)

①人脸识别类FaceRecognition的参数，用于训练模型时使用。

```
参数解释：
Store model:是否保存模型（bool类型，默认值False）
Shuffle data set：是否使用随机洗牌方式划分训练验证集，否则分别从40个不同人脸库里各自按照一定比例划分训练验证集（bool类型，默认值False）
Learning_rate:设置模型的学习率（float类型，默认值0.0001）
Iteration:模型训练迭代次数（int类型，默认值50）
Units：全连接层神经元个数（int类型，默认值2019）
Early stopping：当验证准确率连续下降n次停止训练模型（int类型，默认值5）
Divided_rate:训练集与验证集的比例（float类型，默认值0.9）
```

②	`Start`按钮：启动训练模型    &nbsp;&nbsp;&nbsp;&nbsp;   Reset按钮：重置以上参数为空

③	模型训练中间输出过程展示

④	`Select...`按钮：用于选择测试使用的图片，左边的文本框显示图片文件名称

⑤	左边的`Selected`图片是已选中的图片展示，`Predicted`图片是模型预测出的结果，即属于哪个人，然后从这个人的目录下面选择第一张进行展示，作为对比。

⑥	测试样本预测结果提示

***
### 结果分析
```
通过自己线下的多次调参，得到较好的效果的模型就是上面参数介绍里面的默认参数，即学习率为0.0001，模型训练次数50次，全连接层神经元个数2019，early_stopping为5，数据集划分比例0.9，当Shuffle data set = False时就可以理解为从每个人的10张图片里面选取9张作为训练集，剩余一张作为验证，即训练集包含360个样本，验证集包含40个样本，模型的最高准确率能达到0.925，即有37张图片分类正确，另外3张分类错误。学习率这个参数特别重要，设置过大可能会跨过最优值，太小导致梯度下降太慢，模型训练时间过长且容易过拟合。然后我还尝试了划分比例0.8，0.7，它们的准确率分别为0.5和0.3.分析原因大致如下：由于训练样本太少容易出现欠拟合，另外可能参数设置不到位等等。我所设计的模型里面，各用了两个卷积层和池化层，当然这个层数对模型结果影响也比较大，后期我会去尝试再加上一个或多个卷积层和池化层，看看效果。还有些比较重要的参数，比如说卷积层里面的卷积核和步长这些，考虑到界面和类参数个数，我就没有采样手动设置这些参数，而是采用事先默认设置的。
```

### 界面相关
本系统界面采用的是`PyQt5`做的，`ui-package`里的`MainInterface.ui`就是用`PyQt5`生成的xml文件，所以需要把它转化成`.py`文件。可以使用如下命令：

```python
pyuic5 -o {pyfile} {uifile}     # pyuic5 在PyQt5装好之后就有了
```

推荐一篇在`pycharm`配置PyQt5的教程，写的很详细：https://www.cnblogs.com/BlueSkyyj/p/8398277.html

另外，也可以通过下面的代码将当前目录下的`.ui`文件转化为`.py`文件。
```python
# -*- coding: utf-8 -*-

import sys
import os
import os.path
import MainInterface   # this is the name of ui file 
from PyQt5.QtWidgets import QApplication, QMainWindow

# the dir of ui file.
dir = './'


# 列出目录下的所有UI文件
def listUiFile():
    list = []
    files = os.listdir(dir)
    # print(files)
    for filename in files:
        # print( dir + os.sep + f )
        # print(os.path.splitext(filename))
        if os.path.splitext(filename)[1] == '.ui':
            list.append(filename)
    # print(list)
    return list


# 把扩展名为.ui的文件改成扩展名为.py的文件
def transPyFile(filename):
    return os.path.splitext(filename)[0] + '.py'


# 调用系统命令把UI文件转换成Python文件
def runMain():
    list = listUiFile()
    for uifile in list:
        pyfile = transPyFile(uifile)
        cmd = 'pyuic5 -o {pyfile} {uifile}'.format(pyfile=pyfile, uifile=uifile)
        print(cmd)
        os.system(cmd)    # 系统执行命令


if __name__ == '__main__':
    # runMain()
    # print('Transfer or upgrade ui => py successfully.')

    app = QApplication(sys.argv)
    Main_window = QMainWindow()
    Main_window.setWindowTitle('Face recognition program.')   # don't work
    ui = MainInterface.Ui_MainWindow()
    ui.setupUi(Main_window)
    Main_window.show()
    sys.exit(app.exec_())
```

***
### 文件简介
`face_recognize.py`:人脸识别类，核心模块。

`load_images.py`:加载图片，并将它保存在矩阵中。

`ui-package/MainInterface.py`:界面主程序，除了`自定义功能区`那几行代码外其它的都是PyQt5自动生成的。

`ui-package/MainInterface.ui`:PyQt5生成的xml文件。

`ui-package/main.py`:**系统主函数，只需要运行这一个文件即可。**

**特别提醒：运行之前把所有`.py`文件里的路径改成自己所用的，不然程序会无法运行。界面程序有些小bug，欢迎`issues`交流**

***
### tensorflow保存或加载模型
https://blog.csdn.net/roger_royer/article/details/86520235

***
### 参考文献
- https://www.cnblogs.com/BlueSkyyj/p/8398277.html
- http://www.tensorfly.cn/tfdoc/api_docs/index.html
- https://blog.csdn.net/liuxiao214/article/details/79048136
- https://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/
- http://cs231n.github.io/convolutional-networks/
- https://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/
