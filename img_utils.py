import cv2
import numpy as np
import random

# 将彩色图像转换为灰度图像。灰度图像是一种图像类型，其中每个像素的颜色信息仅由一个单一的亮度或强度值表示，不包括色彩信息。与彩色图像不同，灰度图像不使用全彩色谱，而是使用灰色阴影的不同程度来表示图像。
# 在灰度图像中：
# 黑色通常由值0表示。
# 白色通常由最大强度值表示，例如在8位图像中是255。
# 中间的灰度值表示从黑到白的渐变阴影。
# 灰度图像通常用于那些不需要色彩信息的应用，例如文本识别、边缘检测和许多计算机视觉任务。通过消除颜色，灰度图像可以减少计算复杂性和存储需求。
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def normalize_(image, mean, std):
    image -= mean
    image /= std

#将两个图像按比例混合。
# alpha: 混合比例。
# image1, image2: 要混合的图像。
def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2

#改变图像的光照。
# data_rng: 数据范围或随机生成器。
# image: 要修改的图像。
# alphastd: 控制光照变化量的标准偏差。
# eigval, eigvec: 特征值和特征向量，用于计算光照变化。
def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3, ))
    image += np.dot(eigvec, eigval * alpha)


# 对图像进行颜色抖动，包括亮度、对比度和饱和度的随机变化。
def color_jittering_(data_rng, image):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)

# 改变图像的饱和度。
# data_rng: 数据范围或随机生成器。
# image: 要修改的图像。
# gs: 灰度图像。
# gs_mean: 灰度图像的均值。
# var: 控制饱和度变化量的变量。
def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])

def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha

def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)
    
def crop_image(image, center, size):
    cty, ctx            = center
    height, width       = size
    im_height, im_width = image.shape[0:2]
    cropped_image       = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
    #计算裁剪区域在原始图像中的边界坐标。
    x0, x1 = max(0, ctx - width // 2), min(ctx + width // 2, im_width)
    y0, y1 = max(0, cty - height // 2), min(cty + height // 2, im_height)
    # 计算裁剪区域边界与中心的距离。
    left, right = ctx - x0, x1 - ctx
    top, bottom = cty - y0, y1 - cty
    # 计算裁剪图像的中心坐标。
    cropped_cty, cropped_ctx = height // 2, width // 2
    # 创建裁剪区域的切片。
    y_slice = slice(cropped_cty - top, cropped_cty + bottom)
    x_slice = slice(cropped_ctx - left, cropped_ctx + right)
    cropped_image[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]
    # 创建一个表示裁剪区域边界的数组。
    border = np.array([
       cropped_cty - top,
       cropped_cty + bottom,
       cropped_ctx - left,
       cropped_ctx + right
    ], dtype=np.float32)
    # 计算裁剪区域的偏移量。
    offset = np.array([
        cty - height // 2,
        ctx - width  // 2
    ])

    return cropped_image, border, offset
