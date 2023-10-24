import numpy as np
import cv2
from img_utils import crop_image

# 修剪检测框，确保它们完全位于图像的边界内，并且具有正宽度和高度
def _clip_detections(image, detections):
    detections = detections.copy()
    height, width = image.shape[0:2]
    
    detections[:, 0:detections.shape[1]:2] = np.clip(detections[:, 0:detections.shape[1]:2], 0, width)
    detections[:, 1:detections.shape[1]:2] = np.clip(detections[:, 1:detections.shape[1]:2], 0, height)
    return detections
    ## 使用 NumPy 的 clip 函数将检测框的 x 坐标（即索引 0 和 2 的值）限制在图像宽度的范围内。
    ## 使用 NumPy 的 clip 函数将检测框的 y 坐标（即索引 1 和 3 的值）限制在图像高度的范围内。


def _resize_image(image, detections, size):
    detections = detections.copy()
    height, width = image.shape[0:2]
    new_height, new_width = size

    image = cv2.resize(image, (new_width, new_height))

    height_ratio = new_height / height
    width_ratio = new_width / width
    detections[:, 0:detections.shape[1]:2] *= width_ratio
    detections[:, 1:detections.shape[1]:2] *= height_ratio
    return image, detections

def _full_image_crop(image, detections):
    detections = detections.copy()
    height, width = image.shape[0:2]

    max_hw = max(height, width)
    center = [height // 2, width // 2]
    size = [max_hw, max_hw]

    image, border, offset = crop_image(image, center, size)
    detections[:, 0:detections.shape[1]:2] += border[2]
    detections[:, 1:detections.shape[1]:2] += border[0]
    return image, detections
