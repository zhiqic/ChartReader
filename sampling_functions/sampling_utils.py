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

#参数：
# shape: 高斯滤波器的形状（高和宽）。
# sigma: 高斯函数的标准偏差，用于控制滤波器的宽度。
def gaussian_2d(shape, sigma=1):
    # 对于给定的形状，这将确定滤波器的中心。
    m, n = [(ss - 1.) / 2. for ss in shape]
    # 使用 np.ogrid 创建一个网格，其范围从负中心坐标到正中心坐标。这将产生一个表示从中心到边缘的距离的网格。
    y, x = np.ogrid[-m:m+1,-n:n+1]
    # 将滤波器中非常小的值设置为0。这可以减少不必要的计算，并确保滤波器的有效范围。
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

#参数：
# heatmap: 用于绘制高斯分布的二维数组（热图）。
# center: 高斯分布的中心坐标（x, y）。
# radius: 高斯分布的半径。
# k: 一个可选的乘法因子，用于调整高斯分布的幅度。
def draw_gaussian(heatmap, center, radius, k=1):
    # 计算高斯分布的直径，它等于半径的两倍加1
    diameter = 2 * radius + 1
    # 调用先前定义的 gaussian_2d 函数来生成一个二维高斯滤波器。
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)
    # 解压中心坐标到 x 和 y 变量。
    x, y = center
    # 获取热图的高度和宽度。
    height, width = heatmap.shape[0:2]
    # 通过比较中心坐标和半径与热图的宽度和高度来确定边界。这确保了高斯分布不会超出热图的边界。
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    # 从热图中提取要修改的区域。
    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    # 从高斯滤波器中提取与热图相对应的部分。
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    #  将高斯滤波器应用于热图的选定区域。使用 np.maximum 确保新值不会小于原始热图中的值，并通过乘以因子 k 来调整高斯分布的强度。
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

# 计算高斯分布的半径，以便在检测框中有一个最小的重叠区域。
# det_size: 检测框的大小，表示为 (height, width)。
# min_overlap: 高斯分布与检测框的最小重叠区域。
def gaussian_radius(det_size, min_overlap):
    # 从检测框大小中提取高度和宽度。
    height, width = det_size
    # 高斯半径的计算可以通过求解三个不同的二次方程来完成。每个方程由系数 a, b, 和 c 定义，以及与检测框大小和最小重叠区域有关的参数。
    # 通过使用二次方程的通解公式，可以计算出每个方程的解。
    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 - sq1) / (2 * a1)

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 - sq2) / (2 * a2)

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / (2 * a3)
    # 返回三个解中的最小值作为高斯半径。
    return min(r1, r2, r3)