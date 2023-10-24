import cv2
import numpy as np
import torch
from config import system_configs
from img_utils import crop_image
from .test_utils import _rescale_points

# nnet: 神经网络模型，用于执行测试。
# images: 输入图像，可以是一批图像。
# K: 一个整数，定义了要检测的最大关键点数量。
# kernel: 核大小，通常用于卷积操作。
def kp_decode(nnet, images, K, kernel=3):
    with torch.no_grad():
        detections_tl, detections_br, group_scores = nnet.test([images], K=K, kernel=kernel)
        detections_tl = detections_tl.data.cpu().numpy().transpose((2, 1, 0))
        detections_br = detections_br.data.cpu().numpy().transpose((2, 1, 0))
        return detections_tl, detections_br, group_scores

def kp_grouping(image, db, nnet, debug=False, decode_func=kp_decode, cuda_id=0):
    # 参数初始化
    K = db.configs["top_k"]
    nms_kernel = db.configs["nms_kernel"]

    categories = db.configs["categories"]
    max_per_image = db.configs["max_per_image"]
    
    height, width = image.shape[0:2]

    detections_point_tl = []
    detections_point_br = []
    scale = 1.0
    new_height = int(height * scale)
    new_width  = int(width * scale)
    new_center = np.array([new_height // 2, new_width // 2])

    inp_height = new_height | 127
    inp_width  = new_width  | 127
    images  = np.zeros((1, 3, inp_height, inp_width), dtype=np.float32)
    ratios  = np.zeros((1, 2), dtype=np.float32)
    borders = np.zeros((1, 4), dtype=np.float32)
    sizes   = np.zeros((1, 2), dtype=np.float32)

    out_height, out_width = (inp_height + 1) // 4, (inp_width + 1) // 4
    height_ratio = out_height / inp_height
    width_ratio  = out_width  / inp_width

    resized_image = cv2.resize(image, (new_width, new_height))
    resized_image, border, offset = crop_image(resized_image, new_center, [inp_height, inp_width])
    # 将图像的像素值归一化到[0, 1]范围
    resized_image = resized_image / 255.

    images[0]  = resized_image.transpose((2, 0, 1))
    borders[0] = border
    sizes[0]   = [int(height * scale), int(width * scale)]
    ratios[0]  = [height_ratio, width_ratio]

    if torch.cuda.is_available():
        images = torch.from_numpy(images).cuda(cuda_id)
    else:
        images = torch.from_numpy(images)
        
    # 调用解码函数来获取检测，并重新缩放点以匹配原始图像尺寸。
    dets_tl, dets_br, group_scores = decode_func(nnet, images, K, kernel=nms_kernel)
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
    for j in range(1, categories):
        keep_inds_p = (classes_p_tl == j)
        top_points_tl[j] = detections_point_tl[keep_inds_p].astype(np.float32)
        keep_inds_p = (classes_p_br == j)
        top_points_br[j] = detections_point_br[keep_inds_p].astype(np.float32)
    # 使用NumPy的hstack函数，这些分数数组被水平地堆叠在一起。也就是说，所有不同类别的顶部关键点的分数都被合并成一个一维数组。
    # 例如有三个NumPy数组：array1、array2 和 array3，它们分别是 [1,2,3]、[4,5,6] 和 [7,8,9]。当我们使用 np.hstack([array1, array2, array3])，这些数组会被水平堆叠（横向拼接）成一个新的数组：[1,2,3,4,5,6,7,8,9]。
    scores = np.hstack([
        # 遍历top_points_tl字典中的所有类别（从1到categories + 1）。对于每一个类别j，它取出与该类别对应的所有顶部（左上角）关键点的分数（位于NumPy数组的第0列）。
        top_points_tl[j][:, 0]
        for j in range(1, categories + 1)
    ])
    # 如果检测到的点数超过了最大限制，进行剪裁。
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
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, categories + 1):
            keep_inds = (top_points_br[j][:, 0] >= thresh)
            top_points_br[j] = top_points_br[j][keep_inds]

    return top_points_tl, top_points_br, group_scores

def testing(image, db, nnet, debug=False):
    return globals()[system_configs.testing_function](image, db, nnet, debug=debug)
