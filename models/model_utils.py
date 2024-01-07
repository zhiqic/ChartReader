import torch.nn as nn
from .py_utils import convolution

def make_pool_layer() -> nn.Module:
    return nn.Sequential()

# 参数 kernel: 卷积核的大小。
# 参数 dim0: 输入层的维度。
# 参数 dim1: 输出层的维度。
# 参数 mod: 除了第一个之外要添加的模块数量。
# 参数 layer: 用于构建层的函数，默认为卷积函数。
# 返回类型 nn.Module: 返回一个 PyTorch 模块。
def make_hg_layer(kernel: int, dim0: int, dim1: int, mod: int, layer=convolution, **kwargs) -> nn.Module:
    # Ensure kernel size, dim0 and dim1 are positive and mod is non-negative
    assert kernel > 0 and dim0 > 0 and dim1 > 0, "Kernel size and dimensions must be positive."
    assert mod >= 0, "Number of modules must be non-negative."
    # 使用传入的 layer 函数（默认为卷积函数）创建第一个层，步长为 2。
    # 将此层存储在名为 layers 的列表中。
    layers = [layer(kernel, dim0, dim1, stride=2)]
    # 通过循环，使用相同的 kernel 和 dim1 参数，为列表 layers 添加 mod - 1 个额外的层。
    layers += [layer(kernel, dim1, dim1) for _ in range(mod - 1)]
    
    return nn.Sequential(*layers)