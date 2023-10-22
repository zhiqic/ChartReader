import numpy as np
import cv2
from img_utils import crop_image

# 修剪检测框，确保它们完全位于图像的边界内，并且具有正宽度和高度
def _clip_detections(image, detections):
    clipped_detections = []
    height, width = image.shape[0:2]
    # 使用 NumPy 的 clip 函数将检测框的 x 坐标（即索引 0 和 2 的值）限制在图像宽度的范围内。
    for sublist in detections:
        new_sublist = sublist.copy()  # 创建子列表的副本
        for i in range(0, len(new_sublist), 2):  # 模仿原始NumPy代码的步长为2的切片
            new_sublist[i] = max(0, min(new_sublist[i], width))  # 执行裁剪操作
        for i in range(1, len(new_sublist), 2):
            new_sublist[i] = max(0, min(new_sublist[i], height))  # 执行裁剪操作
        clipped_detections.append(new_sublist)  # 将修改后的子列表添加到 new_detections
    # detections[:, 0:detections.shape[1]:2] = np.clip(detections[:, 0:detections.shape[1]:2], 0, width)
    # 使用 NumPy 的 clip 函数将检测框的 y 坐标（即索引 1 和 3 的值）限制在图像高度的范围内。
    #detections[:, 1:detections.shape[1]:2] = np.clip(detections[:, 1:detections.shape[1]:2], 0, height)
    # 计算一个布尔索引，该索引表示哪些检测框具有正宽度和高度（即右坐标大于左坐标，下坐标大于上坐标）。
    #checked_detections = [sublist for sublist in clipped_detections if (sublist[2] - sublist[0] > 0) and (sublist[3] - sublist[1] > 0)]
    # keep_inds  = ((detections[:, 2] - detections[:, 0]) > 0) & ((detections[:, 3] - detections[:, 1]) > 0)	
    # detections = detections[keep_inds]	
    return clipped_detections
    #return checked_detections

def _resize_image(image, detections, size):
    resized_detections = []
    height, width = image.shape[0:2]
    new_height, new_width = size

    image = cv2.resize(image, (new_width, new_height))

    height_ratio = new_height / height
    width_ratio = new_width / width
    for sublist in detections:
        new_sublist = sublist.copy() 
        for i in range(0, len(new_sublist), 2):  # 模仿原始NumPy代码的步长为2的切片
            new_sublist[i] *= width_ratio
        for i in range(1, len(new_sublist), 2):  # 模仿原始NumPy代码的步长为2的切片
            new_sublist[i] *= height_ratio
        resized_detections.append(new_sublist)
    return image, detections

def _full_image_crop(image, detections):
    detections = detections.copy()
    height, width = image.shape[0:2]

    max_hw = max(height, width)
    center = [height // 2, width // 2]
    size = [max_hw, max_hw]

    image, border, offset = crop_image(image, center, size)
    #detections[:, 0:len(detections):2] += border[2]
    #detections[:, 1:len(detections):2] += border[0]
    cropped_detections = []
    for sublist in detections:
        new_sublist = sublist.copy() 
        for i in range(0, len(new_sublist), 2):  # 模仿原始NumPy代码的步长为2的切片
            new_sublist[i] += border[2]
        for i in range(1, len(new_sublist), 2):  # 模仿原始NumPy代码的步长为2的切片
            new_sublist[i] += border[0]
        cropped_detections.append(new_sublist)
    return image, cropped_detections

# 计算给定尺寸内的有效边界。它确保边界不会太接近尺寸的一半，否则可能会导致某些计算或绘制问题。
# border: 初始边界值。
# size: 边界所在的尺寸（例如宽度或高度）。
def _get_border(border, size):
    # 初始化一个变量 i，该变量用于增加边界的分母。
    i = 1
    # 循环条件检查边界是否太接近尺寸的一半。如果边界除以 i 后接近或超过尺寸的一半，循环将继续。
    while size - border // i <= border // i:
        # 在每次迭代中，将 i 乘以2。这会逐渐增加边界的分母，从而减小边界。
        i *= 2
    # 返回计算出的有效边界。 
    return border // i