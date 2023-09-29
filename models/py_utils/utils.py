import torch.nn as nn

# 上采样层
# 将较低分辨率的数据转换为较高分辨率
class upsample(nn.Module):
    # 输入参数:
    # scale_factor: 上采样的比例因子。如果为2，那么输出的尺寸将是输入的两倍。
    def __init__(self, scale_factor):
        super(upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor)

#合并层，用于相加两个张量。
class merge(nn.Module):
    def forward(self, x, y):
        return x + y

# 卷积层
# 名称源于卷积运算，这是一种数学操作，用于通过一组可学习的权重来混合输入数据的局部区域。
# 工作原理
# 定义卷积核: 卷积核是一个小的权重矩阵，用于在输入数据上进行卷积运算。例如假设我们有一个 5x5 的输入图像和一个 3x3 的卷积核，通过将卷积核应用于输入的每个 3x3 区域并执行卷积运算，我们得到一个 3x3 的输出。
# 滑动窗口: 卷积核在输入数据上滑动，每次移动一定的步长（通常为 1）。在每个位置，卷积核与其覆盖的输入数据的对应元素相乘，然后将结果相加。
# 激活函数: 卷积的输出通常会传递给一个激活函数，例如 ReLU，以引入非线性。
# 可选的偏置项: 可以在卷积后添加一个偏置项，然后再应用激活函数。
# 步长和填充: 步长控制卷积核每次滑动的单位数；填充是在输入周围添加额外的零，以控制输出的空间大小。
# 卷积层能够有效地捕捉输入数据的局部结构和模式。例如，在图像处理中：
# 特征提取: 通过使用不同的卷积核，卷积层可以检测边缘、角点和其他图像特征。
# 参数共享: 同一个卷积核在整个输入上共享，从而减少了模型的参数数量，使其更易于训练。
# 空间不变性: 通过学习在空间上平移不变的特征，卷积层允许模型识别无论其在图像中的位置如何的对象。
class convolution(nn.Module):
    #输入参数:
    # k: 卷积核的大小。
    # inp_dim: 输入特征的维度或通道数。
    # out_dim: 输出特征的维度或通道数。
    # stride: 卷积的步长，默认为 1。
    # with_bn: 是否包括批量归一化层，默认为 True。
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()
        # 计算填充大小，以便卷积后的输出与输入具有相同的空间维度。
        pad = (k - 1) // 2
        # 定义一个卷积层，具有指定的输入维度、输出维度、核大小、填充和步长。如果 with_bn 为 True，则不使用偏置。
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        # 如果 with_bn 为 True，则定义一个批量归一化层。否则，定义一个空的顺序容器。
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        # 定义一个 ReLU 激活函数，其中 inplace=True 表示直接在输入上执行操作，而不是创建新的输出。
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu

# 创建一个由相同类型层组成的序列。
def make_layer(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    #  创建一个层列表，其中的第一层具有指定的输入通道数。这里的 layer 是一个可调用对象（例如，一个卷积层类），它的参数是卷积核的大小、输入通道数、输出通道数以及其他任何关键字参数。
    layers = [layer(k, inp_dim, out_dim, **kwargs)]
    for _ in range(1, modules):
        # 循环创建其余的层。每一层的输入通道数等于上一层的输出通道数。
        layers.append(layer(k, out_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)

# 与 make_layer 函数相比，make_layer_revr 函数的主要区别在于，除了最后一个层外，所有层的输入和输出通道数都保持为 inp_dim。最后一个层的输出通道数为 out_dim。
def make_layer_revr(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    layers = []
    # 循环创建前 modules-1 个层。
    for _ in range(modules - 1):
        # 将新层添加到列表中，其中输入和输出通道数都等于 inp_dim。
        layers.append(layer(k, inp_dim, inp_dim, **kwargs))
    # 创建最后一个层，其中输入通道数为 inp_dim，输出通道数为 out_dim。
    layers.append(layer(k, inp_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)

# 全连接层的主要任务是将前一层的所有激活连接到当前层的每个神经元。
# 线性组合: 每个神经元在全连接层中都与前一层的所有神经元相连。它们的连接由权重组成，这些权重与前一层的输出相乘，然后加上偏置项。
# 激活函数: 线性组合的输出通常被传递给一个激活函数，例如 Sigmoid、ReLU 或 Tanh。这引入了非线性，使网络能够学习更复杂的模式。
class fully_connected(nn.Module):
    def __init__(self, inp_dim, out_dim, with_bn=True):
        super(fully_connected, self).__init__()
        self.with_bn = with_bn

        self.linear = nn.Linear(inp_dim, out_dim)
        if self.with_bn:
            self.bn = nn.BatchNorm1d(out_dim)
        self.relu   = nn.ReLU(inplace=True)

    def forward(self, x):
        linear = self.linear(x)
        bn     = self.bn(linear) if self.with_bn else linear
        relu   = self.relu(bn)
        return relu

# 残差层，通过允许跳过一些层来改进训练过程，有助于解决深度网络中的梯度消失问题。
# 残差层的核心思想是通过学习残差函数，即输入与输出之间的差异，而不是直接学习所需的输出。这是通过以下步骤实现的：
# 残差块: 一个残差块通常由几个卷积层组成，中间可能还有非线性激活函数。
# 跳跃连接: 输入不仅被传递到卷积层，还通过所谓的跳跃连接或短路连接直接与块的输出相加。这就是所谓的“残差”连接，因为卷积层学习的是输入和输出之间的残差或差异。
# 输出: 残差块的输出是卷积层的输出与输入的和。这可以表示为 F(x)+x，其中 F(x) 是卷积层的输出，x 是输入。
class residual(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(residual, self).__init__()
        # 定义了残差块的第一个卷积层、批量归一化（batch normalization）层和激活函数。
        # 批量归一化在训练过程中对每一批数据进行归一化处理。具体来说，它会对每一批数据的每个特征计算均值和方差，并使用这些统计信息来归一化数据。然后，它会使用两个可学习的参数（通常称为比例因子 γ 和偏移因子 β）来缩放和平移归一化后的数据。
        # 加速训练：通过将输入归一化，BN 可以使学习过程更稳定，允许使用更大的学习率，从而加速训练过程。
        # 减轻内部协变量偏移：在训练深度网络时，每一层的参数更新都会导致后续层输入分布的改变。这种现象称为内部协变量偏移，会使训练变得复杂和不稳定。通过对每层的输入进行归一化，BN 能够减轻这一问题。
        # 允许使用饱和激活函数：归一化后的输入落在激活函数的非饱和区域，这有助于减轻梯度消失问题，从而允许使用如 Sigmoid 和 Tanh 等饱和激活函数。
        # 正则化效果：由于在每个批次中计算均值和方差，BN 为模型添加了一些噪声，这具有轻微的正则化效果，有助于防止过拟合。
        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
        self.bn1   = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2   = nn.BatchNorm2d(out_dim)
        # 定义跳跃连接。如果步长不为1或输入和输出通道数不同，则通过1x1卷积来匹配尺寸。 
        self.skip  = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu  = nn.ReLU(inplace=True)

    # 定义了前向传播的过程。首先，输入经过第一个卷积层、批量归一化和激活函数，然后通过第二个卷积层和批量归一化。最后，将跳跃连接的输出加到第二个批量归一化的输出上，并通过 ReLU 激活函数。
    def forward(self, x):
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2   = self.bn2(conv2)

        skip  = self.skip(x)
        return self.relu(bn2 + skip)