# darknet19
Darknet19是一个轻量级的卷积神经网络，用于图像分类和检测任务。Darknet-19的网络结构主要包括卷积层(Convolutional layers)、最大池化层(Max Pooling layers)和全连接层(Fully Connected layers)，并且使用了ReLU激活函数。
本网络中的初始化方法 __init__中定义了五个卷积层
self.layer1 = ConvLayer(3, 32) 创建了第一个卷积层，输入通道数为3（通常是RGB图像），输出通道数为32。这表示网络的第一层将输入的3通道图像转换为32个特征图。
self.layer2 = ConvLayer(32, 64, stride=2) 第二个卷积层，输入通道数为32，输出通道数为64，步长为2。这意味着该层不仅增加了特征图的数量，还会减小特征图的空间尺寸（因为步长大于1）。
self.layer3 = ConvLayer(64, 128) 第三个卷积层，输入通道数为64，输出通道数为128。
self.layer4 = ConvLayer(128, 128) 第四个卷积层，输入通道数和输出通道数都为128。这通常意味着网络在这个阶段试图捕捉更高级别的特征。
self.layer5 = ConvLayer(128, 256, stride=2) 第五个卷积层，输入通道数为128，输出通道数为256，步长为2。同样，这层会进一步减小特征图的尺寸。
forward描述了数据如何从前向后通过网络的各个层：
x = self.layer1(x)：数据首先通过 layer1 层。layer1 是一个 ConvLayer 实例，它会对输入 x 进行卷积、批量归一化和激活函数的操作。
x = self.layer2(x)：接着，输出结果再次作为输入传递给 layer2 层，依此类推。
x = self.layer3(x)：数据继续通过 layer3 层。
x = self.layer4(x)：数据接着通过 layer4 层。
x = self.layer5(x)：数据最后通过 layer5 层。
return x：返回经过所有层处理后的输出。此时的 x 已经经过了多次卷积操作，特征已经被充分提取和变换。
当使用这个网络进行预测时，只需要创建一个 Darknet19 的实例，并传入输入数据即可自动执行 forward 方法，得到网络的输出。
