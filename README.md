# NumPy-CNN

## 实现方法

### 总体说明

使用 Python 与 NumPy 实现了简单的神经网络，称为 `numpy_cnn`。

神经网络的训练过程可以分为输入数据、前向传播、计算损失、反向传播梯度、更新参数几个阶段，而主要的运算则可以分为数据在网络层间的流动、损失的计算、参数的更新这三种。考虑到以上不同类型的运算，将 `numpy_cnn` 分为了以下几个层次组件：

- Tensor 张量：可学习参数使用 Tensor，内部保存参数的值与梯度，底层直接使用 NumPy 的 `ndarray`；数据与梯度也都用 NumPy 的 `ndarray` 表示；
- Layers 网络层：网络层接收输入，进行运算并输出结果；所有的网络层都有前向传播与反向传播运算；
- Loss 损失函数：给定网络输出值与真实值，损失函数计算损失值与最后一层的梯度；
- Net 网络：网络将多个网络层组合起来，实现数据与梯度的流动；
- Optimizer 优化器：完成反向传播得到梯度后，使用优化器对可学习参数进行更新。

基于实现的各个组件，整体的流程为：

```python
# 构建网络、损失函数、优化器
net = Net([layer1, layer2, ...])
loss_f = CrossEntropyLoss()
optimizer = Adam(net.parameters)

# 训练过程
for epoch in range(num_epochs):
    for X, y in data_itetator():
        # 前向传播
        preds = net.forward(X)
        # 计算 loss 与最后一层的梯度
        l = loss_f.loss(preds, y)
        grad = loss_f.grad(preds, y)
        # 梯度反传
        net.backward(grad)
        # 参数更新
        optimizer.update()

# 推断过程
test_preds = net.forward(X_test)
```

### Tensor

我们实现的 Tensor 很简单，就是把两个 `numpy.ndarray` 组合在了一起，分别表示数据与梯度。如前面所说，可学习参数用 Tensor，而 `net.parameters` 就是网络所有的可学习参数组成的列表，前面的示例代码中我们将其传给了优化器。通过这种设计，优化器就可以用 Tensor 内的梯度对参数值进行更新，比较方便。

### 网络层

我们主要实现了2维卷积、2维最大池化、全连接以及 ReLU 激活层。每个网络层都需要实现 `forward` 函数与 `backward` 函数，分别进行前向传播与反向传播的计算，示例代码中网络的前向传播与反向传播实际上就是依次调用各网络层的 `forward` 与 `backward` 函数。

在实现的网络层中，全连接层与卷积层有可学习参数，而池化层与激活层则没有。全连接层中，我们在 `forward` 函数中计算 $wx+b$，其中$x$、$w$、$b$ 分别为输入、权重和偏置，并将计算后的结果输出以便继续传递；而 `backward` 函数则接收来自后一层回传的梯度，并计算其关于 $w$、$b$ 和 $x$ 的梯度，关于 $w$ 和 $b$ 的梯度保存到相应 Tensor 中用于更新参数，而关于 $x$ 的梯度则作为函数的返回值用于继续回传。

卷积层的实现中，除了上一段中提到的内容外，还要考虑 padding 的影响，也就是 `backward` 函数最后输出的应该是关于不进行 padding 的 $x$ 的梯度。此外，由于卷积操作中循环较多，直接计算效率很低，我们使用了著名的 `im2col` 方法，将卷积操作转化成了两个矩阵相乘，这两个矩阵分别对应与输入数据与卷积核。因为numpy对矩阵相乘等运算进行了很多优化，这种转化可以大大提升网络的效率。

池化层与激活层没有可学习参数，它们的 `backward` 函数只需计算后一层回传来的梯度对输入数据的梯度，再将计算得到的梯度输出即可。

在具体实现时，我们也参考了一些网上的博客与代码，如 [NN-by-Numpy](https://github.com/leeroee/NN-by-Numpy)。

### 损失函数

损失函数需要计算损失值与关于预测值的梯度。因为要解决分类问题，我们实现了多分类问题中常用的 Softmax 交叉熵损失函数。

设模型的输出为 $o$，类别数为 $K$，可以使用 Softmax 计算概率分布，结果为

$$\hat{y}_k = \frac{\exp(o_k)}{\sum_{k=1}^K \exp(o_k)}$$

而多分类下的交叉熵损失为

$$J_{CE}(y,\hat{y}) = -\sum_{i=1}^N\sum_{k=1}^K y^{(i)}_k \log(\hat{y}^{(i)}_k)=-\sum_{i=1}^N \log(\hat{y}^{(i)}_c) = -\sum_{i=1}^N (o^{(i)}_c - \log(\sum_{k=1}^K\exp(o^{(i)}_k)))$$

其中 $N$ 为样本数目，$c$ 表示真实类别。再求交叉熵损失 $J_{CE}$ 关于 $o$ 的梯度即可。

### 优化器

示例代码中我们将网络的参数 `net.parameters` 传给了优化器。`net.parameters` 是整个网络所有可学习参数 Tensor 的列表，在网络完成反向传播后，Tensor 中保存着参数的值与梯度，利用 `optimizer.update()` 就对参数进行了更新。

开始时使用了随机梯度下降法 SGD，但网络收敛比较慢。后来参考了 [NN-by-Numpy](https://github.com/leeroee/NN-by-Numpy) 实现的 Adam 算法，收敛速度快了很多。

### 网络参数初始化

全连接层与卷积层的权重参数使用 XavierUniform 方法进行初始化，而偏置参数则初始化为 0。

## 实验与结果分析

分别使用了一个只有全连接层与 ReLU 激活层的全连接网络和一个类似于LeNet 的网络两种模型进行了实验。LeNet的网络结构如图所示。

![LeNet](./figure/lenet.png)

我们的全连接网络使用了三个全连接层，两个激活层，称为 fc-net，其网络定义为：

```Python
net = Net([
    layers.Linear(784, 400),
    layers.ReLU(),
    layers.Linear(400, 100),
    layers.ReLU(),
    layers.Linear(100, 10)
])
```

使用的卷积网络与 LeNet 比较像，但也有些差别，用了两个卷积层，三个全连接层，称为 conv-net，其网络定义为：

```Python
net = Net([
    layers.Conv2d(1, 6, k_size=(5, 5), stride=(1, 1), padding=(2, 2)),
    layers.ReLU(),
    layers.MaxPool2d(k_size=[2, 2], stride=[2, 2]),
    layers.Conv2d(6, 16, k_size=[5, 5], stride=[1, 1]),
    layers.ReLU(),
    layers.MaxPool2d(k_size=[2, 2], stride=[2, 2]),
    layers.Reshape(-1),
    layers.Linear(400, 120),
    layers.ReLU(),
    layers.Linear(120, 84),
    layers.ReLU(),
    layers.Linear(84, 10)
])
```

某次训练中，fc-net 在 14 个 epoch 后，测试集准确率取得最佳结果，为 98.64%；某次训练中，conv-net 在 12 个 epoch 后取得最佳结果，为 99.16%。整体来看，conv-net 的结果优于 fc-net。不过，在我们的机器上，fc-net 训练一个 epoch 只需要 6.2 秒左右，而 conv-net 一个 epoch 需要 76 秒左右。
