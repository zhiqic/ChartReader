import cv2
import numpy as np
import torch
from config import system_configs
from img_utils import crop_image
from .test_utils import _rescale_points

def kp_decode(nnet, images, K, kernel=3):
    # 所有的操作都不会进行梯度计算，从而节省内存和计算时间。
    with torch.no_grad():
            # detections_tl_detections_br: 是一个包含两个部分（detections_tl 和 detections_br）的元组或列表。
            # time_backbone: 是执行backbone网络所需的时间。
            # time_psn: 是执行某个不明确的PSN（可能是某种后处理或网络结构）所需的时间。
            detections_tl_detections_br, time_backbone, time_psn = nnet.test([images], K=K, kernel=kernel)
            detections_tl = detections_tl_detections_br[0]
            detections_br = detections_tl_detections_br[1]
            # 重新排列数组的维度。原始数组的第三维（索引为2）现在变成了新数组的第一维，原始数组的第二维（索引为1）现在变成了新数组的第二维，原始数组的第一维（索引为0）现在变成了新数组的第三维。
            detections_tl = detections_tl.data.cpu().numpy().transpose((2, 1, 0))
            detections_br = detections_br.data.cpu().numpy().transpose((2, 1, 0))
            return detections_tl, detections_br

def kp_detection(image, db, nnet, decode_func=kp_decode, cuda_id=0):
    K = db.configs["top_k"]
    nms_kernel = db.configs["nms_kernel"]

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
    dets_tl, dets_br = decode_func(nnet, images, K, kernel=nms_kernel)

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
    for j in range(1, categories + 1):
        keep_inds_p = (classes_p_tl == j)
        top_points_tl[j] = detections_point_tl[keep_inds_p].astype(np.float32)
        keep_inds_p = (classes_p_br == j)
        top_points_br[j] = detections_point_br[keep_inds_p].astype(np.float32)
    # 从 top_points_tl 的每一个类别中提取分数，并将这些分数水平堆叠在一起，形成一个单一的数组 scores。
    scores = np.hstack([
        # 这个表达式从字典 top_points_tl 中获取键为 j 的数组，并选取该数组的所有行（:）和第一列（索引为 0）。这基本上是获取每个类别 j 的所有分数。
        # 例如，如果 top_points_tl[1] 的分数是 [0.9, 0.8]，top_points_tl[2] 的分数是 [0.7]，那么 scores 将会是 [0.9, 0.8, 0.7]。这样，scores 包含了所有类别的分数，用于后续的操作。
        top_points_tl[j][:, 0]
        for j in range(1, categories + 1)
    ])

    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, categories + 1):
            keep_inds = (top_points_tl[j][:, 0] >= thresh)
            top_points_tl[j] = top_points_tl[j][keep_inds]

    scores = np.hstack([
        top_points_br[j][:, 0]
        for j in range(1, categories + 1)
    ])

    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        # 这行代码使用了 NumPy 的 np.partition 函数，它对数组进行部分排序。这里，kth 是数组 scores 的索引，np.partition 会返回一个新数组，其中索引为 kth 的元素放在了它在排序数组中应处于的位置。
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, categories + 1):
            keep_inds = (top_points_br[j][:, 0] >= thresh)
            top_points_br[j] = top_points_br[j][keep_inds]
    return top_points_tl, top_points_br

def testing(image, db, nnet):
    return globals()[system_configs.testing_function](image, db, nnet)
