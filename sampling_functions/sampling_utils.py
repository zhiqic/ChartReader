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

# 函数从给定的图像中随机裁剪一个区域，并相应地调整检测框
# image: 输入图像。
# detections: 检测框，通常表示为一个矩阵，其中每一行是一个边界框。
# random_scales: 可用于裁剪的随机比例列表。
# view_size: 裁剪视图的尺寸（高度和宽度）。
# border: 裁剪边界的大小。
def random_crop(image, detections, random_scales, view_size, border=64):
    # 从 view_size 和 image 中提取相应的高度和宽度。
    view_height, view_width   = view_size
    image_height, image_width = image.shape[0:2]
    # 从 random_scales 中随机选择一个比例，并计算裁剪的高度和宽度。
    scale  = np.random.choice(random_scales)
    height = int(view_height * scale)
    width  = int(view_width  * scale)
    # 创建一个全零的数组，形状与裁剪尺寸相同，以存储裁剪的图像。
    cropped_image = np.zeros((height, width, 3), dtype=image.dtype)
    # 随机选择裁剪区域的中心，确保它在边界内。
    w_border = _get_border(border, image_width)
    h_border = _get_border(border, image_height)
    # 随机选择裁剪区域的中心，确保它在边界内。
    ctx = np.random.randint(low=w_border, high=image_width - w_border)
    cty = np.random.randint(low=h_border, high=image_height - h_border)
    # 计算裁剪区域的左、右、上、下边界。
    x0, x1 = max(ctx - width // 2, 0),  min(ctx + width // 2, image_width)
    y0, y1 = max(cty - height // 2, 0), min(cty + height // 2, image_height)
    # 计算裁剪图像中心的坐标。
    left_w, right_w = ctx - x0, x1 - ctx
    top_h, bottom_h = cty - y0, y1 - cty

    # crop image
    cropped_ctx, cropped_cty = width // 2, height // 2
    x_slice = slice(cropped_ctx - left_w, cropped_ctx + right_w)
    y_slice = slice(cropped_cty - top_h, cropped_cty + bottom_h)
    cropped_image[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

    # crop detections
    cropped_detections = []
    for sublist in detections:
        new_sublist = sublist.copy() 
        for i in range(0, len(new_sublist), 2):  # 模仿原始NumPy代码的步长为2的切片
            new_sublist[i] -= x0
        for i in range(1, len(new_sublist), 2):  # 模仿原始NumPy代码的步长为2的切片
            new_sublist[i] -= y0
        for i in range(0, len(new_sublist), 2):  # 模仿原始NumPy代码的步长为2的切片
            new_sublist[i] += cropped_ctx - left_w
        for i in range(1, len(new_sublist), 2):
            new_sublist[i] += cropped_cty - top_h
        cropped_detections.append(new_sublist)
    return cropped_image, cropped_detections, scale

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