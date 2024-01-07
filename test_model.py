import cv2
import numpy as np
import torch
from config import system_configs
from img_utils import crop_image

# 根据给定的比例、边界和尺寸重新缩放检测框中的点
# dets: 检测框，通常是一个三维数组，其中每个检测框包括位置信息。
# ratios: 重新缩放的比例，用于调整检测框的大小。
# borders: 边界偏移量，用于调整检测框的位置。
# sizes: 新的尺寸范围，用于限制检测框的大小。
def _rescale_points(dets, ratios, borders, sizes):
    # 从检测框中提取x和y坐标。
    xs, ys = dets[:, :, 2], dets[:, :, 3]
    # 通过除以给定的x方向比例来重新缩放x坐标。
    xs    /= ratios[0, 1]
    # 通过除以给定的y方向比例来重新缩放y坐标。
    ys    /= ratios[0, 0]
    # 将x, y坐标减去边界偏移量，以调整位置。
    xs    -= borders[0, 2]
    ys    -= borders[0, 0]
    # 使用numpy的clip函数将x, y坐标限制在新的尺寸范围内。
    # 如果数组中的元素低于下限，则将其设置为下限值；如果高于上限，则将其设置为上限值。对于在下限和上限之间的值，np.clip 会保留原始值。
    np.clip(xs, 0, sizes[0, 1], out=xs)
    np.clip(ys, 0, sizes[0, 0], out=ys)

def kp_decode_detection(nnet, images):
    detections_tl_detection_br, *_ = nnet.test([images])
    detections_tl = detections_tl_detection_br[0]
    detections_br = detections_tl_detection_br[1]
    # 重新排列数组的维度。原始数组的第三维（索引为2）现在变成了新数组的第一维，原始数组的第二维（索引为1）现在变成了新数组的第二维，原始数组的第一维（索引为0）现在变成了新数组的第三维。
    detections_tl = detections_tl.data.cpu().numpy().transpose((2, 1, 0))
    detections_br = detections_br.data.cpu().numpy().transpose((2, 1, 0))
    return detections_tl, detections_br

def test_kp_detection(image, db, nnet, decode_func=kp_decode_detection, cuda_id=0):

    categories = db.configs["categories"]
    max_per_image = db.configs["max_per_image"]
    height, width = image.shape[0:2]

    detections_point_tl = []
    detections_point_br = []
    center = np.array([height // 2, width // 2])

    inp_height = height | 127
    inp_width  = width  | 127
    images  = np.zeros((1, 3, inp_height, inp_width), dtype=np.float32)
    ratios  = np.zeros((1, 2), dtype=np.float32)
    borders = np.zeros((1, 4), dtype=np.float32)
    sizes   = np.zeros((1, 2), dtype=np.float32)

    out_height, out_width = (inp_height + 1) // 4, (inp_width + 1) // 4
    height_ratio = out_height / inp_height
    width_ratio  = out_width  / inp_width

    resized_image = cv2.resize(image, (width, height))
    resized_image, border, offset = crop_image(resized_image, center, [inp_height, inp_width])

    resized_image = resized_image / 255.

    images[0]  = resized_image.transpose((2, 0, 1))
    borders[0] = border
    sizes[0]   = [height, width]
    ratios[0]  = [height_ratio, width_ratio]

    if torch.cuda.is_available():
        images = torch.from_numpy(images).cuda(cuda_id)
    else:
        images = torch.from_numpy(images)

    # 使用 decode_func 函数进行解码以获取检测结果 
    dets_tl, dets_br = decode_func(nnet, images)

    # 对检测到的点进行重新缩放
    _rescale_points(dets_tl, ratios, borders, sizes)
    _rescale_points(dets_br, ratios, borders, sizes)
    # 合并所有检测点
    detections_point_tl.append(dets_tl)
    detections_point_br.append(dets_br)
    #print(detections_point_tl)
    detections_point_tl = np.concatenate(detections_point_tl, axis=1)
    detections_point_br = np.concatenate(detections_point_br, axis=1)
    #print(detections_point_tl)
    classes_p_tl = detections_point_tl[:, 0, 1]
    classes_p_br = detections_point_br[:, 0, 1]
    # 得到一个一维数组，其中包含了原始三维数组 detections_point_tl 的所有数据点（第一维）在第二维的第一个位置（通常是“类别”或其他特征）和第三维的第二个位置的值。
    # reject detections with negative scores
    keep_inds_p = (detections_point_tl[:, 0, 0] > 0)
    # 这行代码选择了数组 detections_point_tl 中第一维（所有元素），第二维的第一个元素（索引为0），第三维的第一个元素（索引为0）。然后检查这些元素是否大于0。结果是一个布尔数组，用于后续的索引。
    detections_point_tl = detections_point_tl[keep_inds_p, 0]
    # 这里使用之前得到的布尔数组 keep_inds_p 来筛选 detections_point_tl 中的元素。第二维的索引为0，意味着我们只保留第二维的第一个元素。
    classes_p_tl = classes_p_tl[keep_inds_p]

    keep_inds_p = (detections_point_br[:, 0, 0] > 0)
    detections_point_br = detections_point_br[keep_inds_p, 0]
    classes_p_br = classes_p_br[keep_inds_p]
    #print(classes_p_tl)
    top_points_tl = {}
    top_points_br = {}
    for j in range(categories):
        keep_inds_p = (classes_p_tl == j)
        top_points_tl[j] = detections_point_tl[keep_inds_p].astype(np.float32)
        keep_inds_p = (classes_p_br == j)
        top_points_br[j] = detections_point_br[keep_inds_p].astype(np.float32)
    # 从 top_points_tl 的每一个类别中提取分数，并将这些分数水平堆叠在一起，形成一个单一的数组 scores。
    scores = np.hstack([
        # 这个表达式从字典 top_points_tl 中获取键为 j 的数组，并选取该数组的所有行（:）和第一列（索引为 0）。这基本上是获取每个类别 j 的所有分数。
        # 例如，如果 top_points_tl[1] 的分数是 [0.9, 0.8]，top_points_tl[2] 的分数是 [0.7]，那么 scores 将会是 [0.9, 0.8, 0.7]。这样，scores 包含了所有类别的分数，用于后续的操作。
        top_points_tl[j][:, 0]
        for j in range(categories)
    ])

    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(categories):
            keep_inds = (top_points_tl[j][:, 0] >= thresh)
            top_points_tl[j] = top_points_tl[j][keep_inds]

    scores = np.hstack([
        top_points_br[j][:, 0]
        for j in range(categories)
    ])

    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        # 这行代码使用了 NumPy 的 np.partition 函数，它对数组进行部分排序。这里，kth 是数组 scores 的索引，np.partition 会返回一个新数组，其中索引为 kth 的元素放在了它在排序数组中应处于的位置。
        thresh = np.partition(scores, kth)[kth]
        for j in range(categories):
            keep_inds = (top_points_br[j][:, 0] >= thresh)
            top_points_br[j] = top_points_br[j][keep_inds]
    return top_points_tl, top_points_br

# nnet: 神经网络模型，用于执行测试。
# images: 输入图像，可以是一批图像。
# K: 一个整数，定义了要检测的最大关键点数量。
# kernel: 核大小，通常用于卷积操作。
def kp_decode_grouping(nnet, images):
    detections_tl, detections_br, group_scores = nnet.test([images])
    detections_tl = detections_tl.data.cpu().numpy().transpose((2, 1, 0))
    detections_br = detections_br.data.cpu().numpy().transpose((2, 1, 0))
    return detections_tl, detections_br, group_scores

def test_kp_grouping(image, db, nnet, decode_func=kp_decode_grouping, cuda_id=0):
    # 参数初始化

    categories = db.configs["categories"]
    max_per_image = db.configs["max_per_image"]
    
    height, width = image.shape[0:2]

    detections_point_tl = []
    detections_point_br = []
    center = np.array([height // 2, width // 2])

    inp_height = height | 127
    inp_width  = width  | 127
    images  = np.zeros((1, 3, inp_height, inp_width), dtype=np.float32)
    ratios  = np.zeros((1, 2), dtype=np.float32)
    borders = np.zeros((1, 4), dtype=np.float32)
    sizes   = np.zeros((1, 2), dtype=np.float32)

    out_height, out_width = (inp_height + 1) // 4, (inp_width + 1) // 4
    height_ratio = out_height / inp_height
    width_ratio  = out_width  / inp_width

    resized_image = cv2.resize(image, (width, height))
    resized_image, border, offset = crop_image(resized_image, center, [inp_height, inp_width])
    # 将图像的像素值归一化到[0, 1]范围
    resized_image = resized_image / 255.

    images[0]  = resized_image.transpose((2, 0, 1))
    borders[0] = border
    sizes[0]   = [height, width]
    ratios[0]  = [height_ratio, width_ratio]

    if torch.cuda.is_available():
        images = torch.from_numpy(images).cuda(cuda_id)
    else:
        images = torch.from_numpy(images)
        
    # 调用解码函数来获取检测，并重新缩放点以匹配原始图像尺寸。
    dets_tl, dets_br, group_scores = decode_func(nnet, images)
    _rescale_points(dets_tl, ratios, borders, sizes)
    _rescale_points(dets_br, ratios, borders, sizes)
    detections_point_tl.append(dets_tl)
    detections_point_br.append(dets_br)
    detections_point_tl = np.concatenate(detections_point_tl, axis=1)
    detections_point_br = np.concatenate(detections_point_br, axis=1)
    # 获取类别信息和处理负分数。
    classes_p_tl = detections_point_tl[:, 0, 1]
    classes_p_br = detections_point_br[:, 0, 1]

    # reject detections with negative scores
    keep_inds_p = (detections_point_tl[:, 0, 0] > 0)
    detections_point_tl = detections_point_tl[keep_inds_p, 0]
    classes_p_tl = classes_p_tl[keep_inds_p]

    keep_inds_p = (detections_point_br[:, 0, 0] > 0)
    detections_point_br = detections_point_br[keep_inds_p, 0]
    classes_p_br = classes_p_br[keep_inds_p]

    top_points_tl = {}
    top_points_br = {}
    for j in range(categories):
        keep_inds_p = (classes_p_tl == j)
        top_points_tl[j] = detections_point_tl[keep_inds_p].astype(np.float32)
        keep_inds_p = (classes_p_br == j)
        top_points_br[j] = detections_point_br[keep_inds_p].astype(np.float32)
    # 使用NumPy的hstack函数，这些分数数组被水平地堆叠在一起。也就是说，所有不同类别的顶部关键点的分数都被合并成一个一维数组。
    # 例如有三个NumPy数组：array1、array2 和 array3，它们分别是 [1,2,3]、[4,5,6] 和 [7,8,9]。当我们使用 np.hstack([array1, array2, array3])，这些数组会被水平堆叠（横向拼接）成一个新的数组：[1,2,3,4,5,6,7,8,9]。
    scores = np.hstack([
        # 遍历top_points_tl字典中的所有类别（从1到categories + 1）。对于每一个类别j，它取出与该类别对应的所有顶部（左上角）关键点的分数（位于NumPy数组的第0列）。
        top_points_tl[j][:, 0]
        for j in range(categories)
    ])
    # 如果检测到的点数超过了最大限制，进行剪裁。
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(categories):
            keep_inds = (top_points_tl[j][:, 0] >= thresh)
            top_points_tl[j] = top_points_tl[j][keep_inds]

    scores = np.hstack([
        top_points_br[j][:, 0]
        for j in range(categories)
    ])
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(categories):
            keep_inds = (top_points_br[j][:, 0] >= thresh)
            top_points_br[j] = top_points_br[j][keep_inds]

    return top_points_tl, top_points_br, group_scores

def testing(image, db, nnet):
    return globals()[system_configs.testing_function](image, db, nnet)
