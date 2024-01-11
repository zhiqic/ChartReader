import torch
import torch.nn as nn
from .utils import convolution, residual

# 定义了一个非常简单的神经网络模块，其作用是将两个相同形状的张量相加。这是一种常见的操作，用于合并或融合来自不同路径或层的信息。
class MergeUp(nn.Module):
    # 前向传播方法接受两个输入张量up1和up2，并返回它们的和。这两个张量的形状必须相同，以便可以逐元素相加。
    def forward(self, up1, up2):
        return up1 + up2

def make_merge_layer():
    return MergeUp()
# 最大池化是一种下采样技术，用于减小特征映射的尺寸。它通过从每个固定大小的窗口中选择最大值，并将其作为新的特征映射的值来工作。这个特定的最大池化层使用2x2的窗口大小和步长2。
# 参数
# kernel_size=2：窗口大小为2x2。
# stride=2：窗口在每个方向上移动的步长为2。
# 作用
# 最大池化层的主要作用包括：
# 减小尺寸：通过减小特征映射的高度和宽度，降低了后续层的计算复杂性。
# 增加感受野：增加了网络对输入图像中更大区域的感知能力。
# 不变性：通过选择窗口中的最大值，最大池化提供了对小的平移、旋转和缩放的不变性。
def make_pool_layer():
    return nn.MaxPool2d(kernel_size=2, stride=2)

# 上采样是一种增加特征映射尺寸的技术。它被用于将低分辨率特征映射转换为高分辨率，从而能够在后续层中捕获更精细的信息。这个特定的上采样层使用了比例因子2，这意味着它将输入特征映射的高度和宽度放大2倍。

# 参数
# scale_factor=2：放大因子为2，意味着在每个方向上都将尺寸加倍。
# 作用
# 上采样层的主要作用包括：

# 增加尺寸：通过将特征映射的高度和宽度增加一倍，为后续层提供更高分辨率的输入。
# 恢复细节：在一系列下采样操作后，上采样可以帮助恢复一些丢失的空间细节。
# 多尺度特征组合：上采样常用于深度学习中的多尺度特征组合，允许网络同时捕获不同分辨率下的图像特征。
def make_unpool_layer(dim):
    return nn.Upsample(scale_factor=2)

# 定义了一个关键点层，用于从给定的输入特征中预测关键点
# 接受三个参数：cnv_dim（输入维度）、curr_dim（中间层的维度）和out_dim（输出维度），并返回一个包含两个卷积层的顺序模块。
def make_kp_layer(cnv_dim, curr_dim, out_dim):
    return nn.Sequential(
        # 特征转换：通过3x3卷积层，对输入特征进行空间转换。
        convolution(3, cnv_dim, curr_dim, with_bn=False),
        # 维度调整：通过1x1卷积层，将特征映射的维度调整为所需的输出维度。这可以用于预测关键点的热图或其他任务特定的输出。
        nn.Conv2d(curr_dim, out_dim, (1, 1))
    )

# 接受一个参数dim，表示输入和输出的维度，然后返回一个3x3的残差块，其输入和输出维度都是dim。
def make_inter_layer(dim):
    return residual(3, dim, dim)

def make_cnv_layer(inp_dim, out_dim):
    return convolution(3, inp_dim, out_dim)

# 根据索引收集特征张量中的特定元素。如果提供了掩码，它还会根据掩码进行筛选
# 输入参数:
# feat: 这是一个特征张量，通常具有形状 (batch_size, num_features, dim)，其中 dim 是特征的维数。
# ind: 这是一个索引张量，通常具有形状 (batch_size, num_indices)，用于从特征张量中选择特定的特征。
# mask (可选): 这是一个可选的掩码张量，用于进一步筛选收集的特征。
def _gather_feat(feat, ind, mask=None):
    # 获取特征张量的第三个维度的大小，即特征的维数，并将其存储在变量 dim 中。
    #print("Gathering feat")
    dim  = feat.size(2)
    #print(f"dim = {dim}") 
    # 使用 unsqueeze(2) 在索引张量的第三个维度上添加一个额外的维度，使其形状变为 (batch_size, num_indices, 1)。
    # 使用 expand 方法将其展开为与特征张量的第三个维度相同的大小，从而得到形状为 (batch_size, num_indices, dim) 的张量。
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    # 检查 ind 张量中的最小和最大值
    min_val = torch.min(ind)
    max_val = torch.max(ind)

    # 获取 feat 张量的第一个维度的大小
    max_index_feat = feat.size(1) - 1  # 因为索引是从 0 开始的

    # 检查是否所有的索引都在有效范围内
    if min_val < 0 or max_val > max_index_feat:
        print(f"Index out of bounds! Min index: {min_val}, Max index: {max_val}, Allowed index range: [0, {max_index_feat}]")
    #print(f"ind = {ind}") 
    # 沿着第一个维度收集特征。在这里，ind 用于从 feat 中选择特定的特征。结果存储在 feat 中。
    #a = torch.tensor([[1, 2], [3, 4], [5, 6]])
    #index = torch.tensor([[0, 0], [2, 1]])
    #output = torch.gather(a, 0, index)
    # 我们用 dim=0 作为参数，这意味着我们在行维度（第 0 维）上进行 gather 操作。
    # 对于输出张量的第一行，我们使用 index 的第一行 [0, 0]。这意味着我们从 a 的第 0 行中取出每一列的元素。也就是取 a[0][0] 和 a[0][1]，它们分别是 1 和 2。
    # 对于输出张量的第二行，我们使用 index 的第二行 [2, 1]。这意味着：
    # 第一个元素是 a 的第三行第一列的元素，即 a[2][0] = 5。
    # 第二个元素是 a 的第二行第二列的元素，即 a[1][1] = 4。
    feat = feat.gather(1, ind)
    if mask is not None:
        # 如果提供了掩码 (mask)，则通过 unsqueeze 和 expand_as 调整其形状以匹配 feat。
        mask = mask.unsqueeze(2).expand_as(feat)
        # 使用掩码从 feat 中选择特定的元素，并通过 view 将结果重新整形为 (-1, dim) 形状。
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

#执行非极大值抑制（Non-Maximum Suppression，NMS），找出热力图中的局部最大值，并将非局部最大值设置为零。NMS是一种用于消除重叠检测框的技术，通过保留每个邻域中的最大值并抑制其他值来实现。
# 输入参数:
# heat: 这是一个热力图张量，通常具有形状 (batch_size, channels, height, width)。
# kernel: 这是一个整数，表示用于 max pooling 操作的核大小，默认值为 1。
def _nms(heat, kernel=1):
    # 计算填充大小 pad。这是为了确保 max pooling 操作后的输出大小与原始热力图相同。
    pad = (kernel - 1) // 2
    # 使用 PyTorch 的 max_pool2d 函数对热力图进行最大池化操作。
    # 核大小由 kernel 参数确定，并且通过设置步长为 1 和填充为 pad 来确保输出与输入热力图的大小相同。
    # 池化操作后的结果存储在 hmax 中。
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    # 通过比较池化后的热力图 hmax 和原始热力图 heat，找出局部最大值的位置。
    # 这个比较会生成一个布尔张量，其中局部最大值的位置为 True，其余位置为 False。
    # 使用 .float() 将布尔张量转换为浮点张量，局部最大值的位置为 1.0，其余位置为 0.0。这个张量存储在 keep 中。
    keep = (hmax == heat).float()
    # 将原始热力图 heat 与 keep 张量相乘，从而保留局部最大值并抑制非局部最大值。
    # 结果返回为新的热力图，其中非局部最大值已被抑制。
    return heat * keep

#首先对特征张量进行转置，然后调用 _gather_feat 收集特定元素。
#feat: 这是一个特征张量，通常具有形状 (batch_size, channels, height, width)。
#ind: 这是一个索引张量，用于从特征张量中选择特定的特征。
def _transpose_and_gather_feat(feat, ind):
    # 使用 permute 方法重新排列特征张量的维度。新的维度顺序为 (batch_size, height, width, channels)。
    # contiguous 方法确保张量在内存中是连续的，这在之后的 view 操作中可能是必需的。
    feat = feat.permute(0, 2, 3, 1).contiguous()
    # 使用 view 方法将特征张量重新整形。新的形状为 (batch_size, height * width, channels)。
    # 这里，-1 表示该维度的大小由其他维度的大小自动推断。
    feat = feat.view(feat.size(0), -1, feat.size(3))
    # 调用之前定义的 _gather_feat 函数，使用索引张量 ind 从重新整形的特征张量中收集特定的特征。
    feat = _gather_feat(feat, ind)
    return feat

#从得分张量中提取前K个最高得分，返回得分及其对应的索引和类别。
#输入参数:
# scores: 得分张量，通常具有形状 (batch_size, num_categories, height, width)，其中每个元素表示特定类别和位置的得分。
# K: 要选择的每个类别的顶部得分的数量。
def _topk(scores, K=20):
    # 从得分张量中获取各个维度的大小。
    batch, cat, height, width = scores.size()
    # 使用 view 方法将得分张量重塑为形状 (batch_size, -1)，其中 -1 表示自动推断该维度的大小。
    # 使用 PyTorch 的 torch.topk 函数获取每个批次中前 K 个得分及其对应的索引。这些存储在 topk_scores 和 topk_inds 中。
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)
    # 计算每个前 K 个得分对应的类别索引。通过将索引除以每个类别的元素数量（即 height * width），可以确定每个得分对应的类别。
    topk_clses = (topk_inds / (height * width)).int()
    # 计算每个前 K 个得分在其对应类别内的相对索引。
    topk_inds = topk_inds % (height * width)
    # 计算每个前 K 个得分的垂直（行）位置。通过将相对索引除以宽度并转换为浮点数，可以得到这个位置
    topk_ys   = (topk_inds / width).int().float()
    # 计算每个前 K 个得分的水平（列）位置。通过取相对索引与宽度的模并转换为浮点数，可以得到这个位置
    topk_xs   = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

def _decode_detection_or_group(tl_heat, br_heat, tl_regr, br_regr, K=100, kernel=1, option="pure"):
    batch, cat, height, width = tl_heat.size()
    
    # 通过Sigmoid激活函数将热图的值限制在0到1之间。
    tl_heat = torch.sigmoid(tl_heat)
    br_heat = torch.sigmoid(br_heat)
    # 非极大值抑制（NMS）用于去除冗余和重叠的检测。这里应用于两个热图。
    tl_heat = _nms(tl_heat, kernel=kernel)
    br_heat = _nms(br_heat, kernel=kernel)
    # 使用_topk函数从每个热图中提取最高分数的K个检测。
    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)
    tl_regr_ = _transpose_and_gather_feat(tl_regr, tl_inds)
    br_regr_ = _transpose_and_gather_feat(br_regr, br_inds)

    # 使用收集的回归特征精细调整检测的位置。
    # view方法用于改变张量的形状。它接收新形状的尺寸作为输入，并返回新形状的张量，其中的数据与原始张量相同。
    # 这里，tl_scores是一个张量，其中包含每个批次的前K个顶部左侧检测的分数。通过调用view(1, batch, K)，我们将其形状更改为(1, batch, K)，其中batch是批次大小，K是每个批次的检测数量。
    tl_scores_ = tl_scores.view(1, batch, K)
    tl_clses_ = tl_clses.view(1, batch, K)
    tl_xs_ = tl_xs.view(1, batch, K)
    tl_ys_ = tl_ys.view(1, batch, K)
    tl_regr_ = tl_regr_.view(1, batch, K, 2)
    # 将x方向的调整添加到检测的x坐标上。
    tl_xs_ += tl_regr_[:, :, :, 0]
    # 将y方向的调整添加到检测的y坐标上。
    tl_ys_ += tl_regr_[:, :, :, 1]
    
    br_scores_ = br_scores.view(1, batch, K)
    br_clses_ = br_clses.view(1, batch, K)
    br_xs_ = br_xs.view(1, batch, K)
    br_ys_ = br_ys.view(1, batch, K)
    br_regr_ = br_regr_.view(1, batch, K, 2)
    br_xs_ += br_regr_[:, :, :, 0]
    br_ys_ += br_regr_[:, :, :, 1]
    # 通过将分数、类别和位置连接在一起，创建顶部左侧和底部右侧的检测。
    detections_tl = torch.cat([tl_scores_, tl_clses_.float(), tl_xs_, tl_ys_], dim=0)
    detections_br = torch.cat([br_scores_, br_clses_.float(), br_xs_, br_ys_], dim=0)
    if (option == "pure"):
        return detections_tl, detections_br
    else:
        return detections_tl, detections_br, tl_inds, br_inds, tl_scores, br_scores_
    
def _decode_detection(
        tl_heat, br_heat, tl_regr, br_regr,
        K=100, kernel=1
):
    return _decode_detection_or_group(tl_heat, br_heat, tl_regr, br_regr,
        K, kernel, "pure")

def _decode_group(
        tl_heat, br_heat, tl_regr, br_regr,
        K=100, kernel=1
):
    return _decode_detection_or_group(tl_heat, br_heat, tl_regr, br_regr,
        K, kernel, "group")

# 这个函数定义了一个负采样损失（negative loss），通常用于二分类问题，特别是在目标检测和图像分割任务中，其中正样本（目标）和负样本（背景）的数量可能会极度不平衡。
# 接受四个参数：preds（模型的预测值），gt（真实的地面真值标签），lambda_ 和 lambda_b（损失计算中使用的超参数）。
def _neg_loss(preds, gt, lambda_, lambda_b):
    # 找到地面真值标签中所有正样本（目标）的索引。
    pos_inds = gt.eq(1)
    # 找到地面真值标签中所有负样本（背景）的索引。
    neg_inds = gt.lt(1)
    # 计算负样本的权重，通常用于平衡正样本和负样本之间的不平衡。
    neg_weights = torch.pow(1 - gt[neg_inds], lambda_)
    loss = 0
    for pred in preds:
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]
        # 计算正样本的损失，使用对数损失和权重。
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, lambda_b)
        # 计算负样本的损失，使用对数损失、权重和先前计算的负权重。
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, lambda_b) * neg_weights
        # 计算正样本的数量，并对正样本和负样本的损失求和。
        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        # 如果没有正样本，则只计算负样本的损失。否则，将正样本和负样本的损失组合，并除以正样本的数量。
        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def _sigmoid(x):
    # x.sigmoid_(): 这里，Sigmoid激活函数在张量 x 上进行了原地（in-place）操作。Sigmoid函数，用于将输入值压缩到 (0,1) 范围内。
    # torch.clamp(...): 使用 torch.clamp 函数将Sigmoid函数的输出限制在范围内。这可以防止数值稳定性问题，特别是在后续可能涉及对数运算的情况下。
    x = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return x

def _regr_loss(regr, gt_regr, mask):
    # 将掩码转换为浮点数，并计算有效元素的数量。
    num = mask.float().sum()
    # 通过添加一个新的维度并扩展掩码，使其与目标回归张量的形状相匹配。
    mask = mask.unsqueeze(2).expand_as(gt_regr)
    # 使用掩码选择那些应该被考虑在损失计算中的预测和真实回归值。
    regr = regr[mask]
    gt_regr = gt_regr[mask]
    # 将总损失除以有效元素的数量，以计算平均损失。为了避免除以零的情况，分母中添加了一个小的常数 
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, reduction='sum')
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss
