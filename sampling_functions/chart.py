import cv2
import numpy as np
import torch
import os
import math
from config import system_configs
from img_utils import color_jittering_, lighting_
from .sampling_utils import draw_gaussian, gaussian_radius, _full_image_crop, random_crop, _resize_image, _clip_detections

def bad_p(x, y, output_size):
    # 检查坐标是否位于输出大小的有效范围之外
    # 通过减去一个非常小的值，该函数确保坐标不会正好位于边界上。
    return x == 0 or y == 0 or x >= (output_size[1]-1e-2) or y >= (output_size[0]-1e-2)

# 计算三个点 a, b, 和 c 所构成的三角形的中心位置
def get_center(a, b, c):
    # 计算从点 a 到点 c 的向量
    ca = [c[0]-a[0], c[1]-a[1]]
    # 计算从点 b 到点 c 的向量
    cb = [c[0]-b[0], c[1]-b[1]]
    # 叉积 ca*cb 的符号表示向量 ca 和 cb 之间的角度的方向
    if ca[0]*cb[1]-ca[1]*cb[0] >= 0:
        # 如果角度为非负值，返回三角形的重心，即三个顶点的坐标平均值
        return (a[0]+b[0]+c[0])/3., (a[1]+b[1]+c[1])/3.
    else:
        # 否则，返回另一个点
        return 2*c[0]-(a[0]+b[0]+c[0])/3., 2*c[1]-(a[1]+b[1]+c[1])/3.


def kp_detection(db, k_ind, debug):
    data_rng   = system_configs.data_rng
    batch_size = system_configs.batch_size

    categories   = 10 
    input_size   = db.configs["input_size"]
    output_size  = db.configs["output_sizes"][0]

    border        = db.configs["border"]
    lighting      = db.configs["lighting"]
    rand_crop     = db.configs["rand_crop"]
    rand_color    = db.configs["rand_color"]
    rand_scales   = db.configs["rand_scales"]
    gaussian_bump = db.configs["gaussian_bump"]
    gaussian_iou  = db.configs["gaussian_iou"]
    gaussian_rad  = db.configs["gaussian_radius"]
    
    # if "Pie" or "Bar"
        # max_tag_len = 128
    # elif "Line"
    max_tag_len = 256
    max_group_len = 16
     # allocating memory
    images          = np.zeros((batch_size, 3, input_size[0], input_size[1]), dtype=np.float32)
    # 分配两个张量，用于存储关键点和中心点的热图
    # 热图的尺寸：通常与输入图像的尺寸不同，因为网络中的卷积和池化操作会改变特征图的大小。在你提供的代码中，output_size（例如 [128, 128]）指定了热图的尺寸。
    # 多类别问题：在多目标检测或多关键点检测任务中，通常为每个类别生成一个独立的热图。在你的代码中，categories 代表类别数，batch_size 是批量大小。
    center_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    key_heatmaps    = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    # 分配两个张量，用于存储关键点和中心点的回归目标
    center_regrs    = np.zeros((batch_size, max_tag_len + 1, 2), dtype=np.float32)
    key_regrs       = np.zeros((batch_size, max_tag_len + 1, 2), dtype=np.float32)
    # 分配两个张量，用于存储关键点和中心点的坐标信息
    center_tags     = np.zeros((batch_size, max_tag_len + 1), dtype=np.int64) # location values
    key_tags        = np.zeros((batch_size, max_tag_len + 1), dtype=np.int64) # location values
    # 分配两个布尔张量，用于存储关键点和中心点的掩码
    key_masks       = np.zeros((batch_size, max_tag_len + 1), dtype=bool)
    center_masks    = np.zeros((batch_size, max_tag_len + 1), dtype=bool)
    # 分配两个一维张量，用于存储每个样本的标签长度
    tag_lens_keys   = np.zeros((batch_size, ), dtype=np.int32)
    tag_lens_cens   = np.zeros((batch_size, ), dtype=np.int32)
    # 分配一个张量，用于存储分组目标
    group_target    = np.zeros((batch_size, max_tag_len + 1, max_tag_len + 1), dtype=np.int64)
    
        
    db_size = db.db_inds.size
    # 在一个批次中选择一个有效的数据点（或多个数据点）。它首先会随机洗牌数据库（如果满足条件），然后使用 while 循环来找到一个有效的数据点
    # k_ind 是一个控制变量，用于追踪我们当前在数据库中的哪个位置
    for b_ind in range(batch_size):
        #print(f"b_ind = {b_ind}")
        if not debug and k_ind == 0:
            db.shuffle_inds()
        flag = False
        while not flag:
            db_ind = db.db_inds[k_ind]
            k_ind = (k_ind + 1) % db_size
            # reading image 
            image_file = db.image_file(db_ind)
            if(os.path.exists(image_file) and len(db.detections(db_ind)) <= max_tag_len//3) and len(db.detections(db_ind)) > 0: 
                (detections, categories) = db.detections(db_ind)
                detections = list(detections)
                categories = list(categories)
                if(len(categories)):
                        image = cv2.imread(image_file)
                        flag = True
        image = cv2.imread(image_file)
        ori_size = image.shape
            #print(temp)
        #print(f"k_ind: {k_ind}")
        (detections, categories) = db.detections(db_ind)
        detections = list(detections)
        categories = list(categories)
        #print(f"Detections: {detections}")
        #print(f"Length of detection: {len(detections)}")
        #print(f"Categories: {categories}")
        for i in range(len(detections)):
                detection = detections[i]
                #print(f"Category: {int(categories[i])}")
                if(categories[i] == 3):
                    if len(detection) < 5:
                        print("Insufficient elements in the detection list.")
                        print(len(detection))
                        print(image_file)
                        continue
                    xce, yce = get_center((detection[0], detection[1]), (detection[2], detection[3]), (detection[4], detection[5]))
                    detections[i] = detection[:6] + [xce, yce] + [detection[-1]]
        if(categories[0] == 2):
            detections = detections[0:max_group_len]
            categories = categories[0:max_group_len]
        # cropping an image randomly
        if not debug and rand_crop:
            image, detections, scale = random_crop(image, detections, rand_scales, input_size, border=border)
        else:
            image, detections = _full_image_crop(image, detections)
            scale = 1
        #print(f"Cropped detections: {detections}")
        image, detections = _resize_image(image, detections, input_size)
        #print(f"Resized detections: {detections}")
        detections = _clip_detections(image, detections)
        width_ratio  = output_size[1] / input_size[1]
        height_ratio = output_size[0] / input_size[0]
        #print(f"width ratio: {width_ratio}, height ratio: {height_ratio}")
        #print(f"Clipped detections: {detections}")
        if not debug:
            image = image.astype(np.float32) / 255.
            if rand_color:
                color_jittering_(data_rng, image)
                if lighting:
                    lighting_(data_rng, image, 0.1, db.eig_val, db.eig_vec)
            #normalize_(image, db.mean, db.std)
        images[b_ind] = image.transpose((2, 0, 1))
        for ind, (detection, category) in enumerate(zip(detections, categories)):
       #     print(f"ind: {ind}, detection: {detection}, category: {category}")
            if(int(category) == 2):
                # remove cropped points
                tmp = []
                for k in range(int(len(detection) / 2)):
                    #print(f"k = {k}")
                    if not bad_p(detection[2*k], detection[2*k+1], input_size):
                        tmp.append(detection[2*k])
                        tmp.append(detection[2*k+1])
                detection = np.array(tmp)

                # get center
                if len(detection) == 0: continue
                elif len(detection)//2 % 2 == 0:
                    mid = len(detection) // 2
                    xce, yce = (detection[mid-2] + detection[mid]) / 2, (detection[mid-1] + detection[mid+1]) / 2
                else:
                    mid = len(detection) // 2
                    xce, yce = detection[mid-1].copy(), detection[mid].copy()
                fxce = (xce * width_ratio)
                fyce = (yce * height_ratio)
                xce = int(fxce)
                yce = int(fyce)
                xce = min(xce, key_heatmaps.shape[3] - 1)
                yce = min(yce, key_heatmaps.shape[2] - 1)
                # get keypoints
                fdetection = detection.copy()
                fdetection[0:len(fdetection):2] = detection[0:len(detection):2] * width_ratio
                fdetection[1:len(fdetection):2] = detection[1:len(detection):2] * height_ratio
                detection = fdetection.astype(np.int32)

                if gaussian_bump:
                    width = ori_size[1] / 50 / 4 / scale
                    height = ori_size[0] / 50 / 4 / scale

                    if gaussian_rad == -1:
                        radius = gaussian_radius((height, width), gaussian_iou)
                        radius = max(0, int(radius))
                    else:
                        radius = gaussian_rad

                    for k in range(int(len(detection) / 2)):
                        if not bad_p(detection[2*k], detection[2*k+1], output_size):
                            draw_gaussian(key_heatmaps[b_ind, int(category)], [detection[2 * k], detection[2 * k + 1]], radius)
                    if not bad_p(xce, yce, output_size):
                        draw_gaussian(center_heatmaps[b_ind, int(category)], [xce, yce], radius)

                else:
                    for k in range(int(len(detection) / 2)):
                        if not bad_p(detection[2*k], detection[2*k+1], output_size):
                            #print(f"k: {k}")
                            #print(f"{detection[2*k + 1]}")
                            #print(f"{detection[2*k]}")
                            key_heatmaps[b_ind, int(category), min(detection[2 * k + 1], key_heatmaps.shape[2] - 1), min(detection[2 * k], key_heatmaps.shape[3] - 1)] = 1
                            center_heatmaps[b_ind, int(category), yce, xce] = 1

                for k in range(int(len(detection) / 2)):
                    if not bad_p(detection[2*k], detection[2*k+1], output_size):
                        #print(f"tag_lens_keys[{b_ind}]: {tag_lens_keys[b_ind]}")
                        if tag_lens_keys[b_ind] >= max_tag_len - 3:
                            print("Too many targets, skip!")
                            print(tag_lens_keys[b_ind])
                            print(image_file)
                            break
                        tag_ind = tag_lens_keys[b_ind]
                        key_regrs[b_ind, tag_ind, :] = [fdetection[2 * k] - detection[2 * k],fdetection[2 * k + 1] - detection[2 * k + 1]]
                        key_tags[b_ind, tag_ind] = detection[2 * k + 1] * output_size[1] + detection[2 * k]
                        group_target[b_ind, tag_lens_cens[b_ind], tag_lens_keys[b_ind]] = 1
                        tag_lens_keys[b_ind] += 1

                if not bad_p(xce, yce, output_size):
                    tag_ind_center = tag_lens_cens[b_ind]
                    center_regrs[b_ind, tag_ind_center, :] = [fxce - xce, fyce - yce]
                    center_tags[b_ind, tag_ind_center] = yce * output_size[1] + xce
                    tag_lens_cens[b_ind] += 1
                else:
                    group_target[b_ind, tag_lens_cens[b_ind], :] = 0

                tag_len = tag_lens_keys[b_ind]
                key_masks[b_ind, :tag_len] = 1
                tag_len = tag_lens_cens[b_ind]
                center_masks[b_ind, :tag_len] = 1
            elif(int(category) == 3):
                # print(f"Processing pies, detection is {detection}")
                if len(detection) < 5:
                        print("Insufficient elements in the detection list.")
                        print(len(detection))
                        print(image_file)
                        continue
                xk1, yk1 = detection[0], detection[1] # arc point 1
                xk2, yk2 = detection[2], detection[3] # arc point 2
                xk3, yk3 = detection[4], detection[5] # center point
                xce, yce = detection[6], detection[7] # center of pie
                # compute the center of mass
                # xce, yce = get_center((xk1, yk1), (xk2, yk2), (xk3, yk3))
                # xce, yce = np.clip(xce, 0, output_size[1] - 1), np.clip(yce, 0, output_size[0] - 1)

                fxk1 = (xk1 * width_ratio)
                fyk1 = (yk1 * height_ratio)
                fxk2 = (xk2 * width_ratio)
                fyk2 = (yk2 * height_ratio)
                fxk3 = (xk3 * width_ratio)
                fyk3 = (yk3 * height_ratio)
                fxce = (xce * width_ratio)
                fyce = (yce * height_ratio)
                xk1 = int(fxk1)
                yk1 = int(fyk1)
                xk2 = int(fxk2)
                yk2 = int(fyk2)
                xk3 = int(fxk3)
                yk3 = int(fyk3)
                xce = int(fxce)
                yce = int(fyce)
                xk1 = min(xk1, key_heatmaps.shape[3] - 1)
                yk1 = min(yk1, key_heatmaps.shape[2] - 1)
                xk2 = min(xk2, key_heatmaps.shape[3] - 1)
                yk2 = min(yk2, key_heatmaps.shape[2] - 1)
                xk3 = min(xk3, key_heatmaps.shape[3] - 1)
                yk3 = min(yk3, key_heatmaps.shape[2] - 1)
                xce = min(xce, key_heatmaps.shape[3] - 1)
                yce = min(yce, key_heatmaps.shape[2] - 1)

                if gaussian_bump:
                    width = math.sqrt(math.pow(xk3-xk1, 2)+math.pow(yk3-yk1, 2))
                    height = math.sqrt(math.pow(xk3-xk2, 2)+math.pow(yk3-yk2, 2))

                    if gaussian_rad == -1:
                        radius = gaussian_radius((height, width), gaussian_iou)
                        radius = max(0, int(radius))
                    else:
                        radius = gaussian_rad

                    draw_gaussian(center_heatmaps[b_ind, int(category)], [xce, yce], radius)
                    draw_gaussian(key_heatmaps[b_ind, int(category)], [xk1, yk1], radius)
                    draw_gaussian(key_heatmaps[b_ind, int(category)], [xk2, yk2], radius)
                    draw_gaussian(key_heatmaps[b_ind, int(category)], [xk3, yk3], radius)

                else:
                    center_heatmaps[b_ind, int(category), yce, xce] = 1
                    key_heatmaps[b_ind, int(category), yk1, xk1] = 1
                    key_heatmaps[b_ind, int(category), yk2, xk2] = 1
                    key_heatmaps[b_ind, int(category), yk3, xk3] = 1

                key_regrs[b_ind, tag_lens_keys[b_ind], :] = [fxk1 - xk1, fyk1 - yk1]
                key_tags[b_ind, tag_lens_keys[b_ind]] = yk1 * output_size[1] + xk1
                group_target[b_ind, tag_lens_cens[b_ind], tag_lens_keys[b_ind]] = 1
                tag_lens_keys[b_ind] += 1
                key_regrs[b_ind, tag_lens_keys[b_ind], :] = [fxk2 - xk2, fyk2 - yk2]
                key_tags[b_ind, tag_lens_keys[b_ind]] = yk2 * output_size[1] + xk2
                group_target[b_ind, tag_lens_cens[b_ind], tag_lens_keys[b_ind]] = 1
                tag_lens_keys[b_ind] += 1
                key_regrs[b_ind, tag_lens_keys[b_ind], :] = [fxk3 - xk3, fyk3 - yk3]
                key_tags[b_ind, tag_lens_keys[b_ind]] = yk3 * output_size[1] + xk3
                group_target[b_ind, tag_lens_cens[b_ind], tag_lens_keys[b_ind]] = 1
                tag_lens_keys[b_ind] += 1
                center_regrs[b_ind, tag_lens_cens[b_ind], :] = [fxce - xce, fyce - yce]
                center_tags[b_ind, tag_lens_cens[b_ind]] = yce * output_size[1] + xce
                tag_lens_cens[b_ind] += 1



                if tag_lens_keys[b_ind] >= max_tag_len-3:
                    #print("Too many targets, skip!")
                    #print(tag_lens_keys[b_ind])
                    #print(image_file)
                    break

                center_masks[b_ind, :tag_lens_cens[b_ind]] = 1
                key_masks[b_ind, :tag_lens_keys[b_ind]] = 1
            else:
                # 提取检测框的左上角和右下角坐标，以及中心点坐标。
                xk1, yk1 = detection[0], detection[1] # top left point	
                xk2, yk2 = detection[2], detection[3] # bottom right point	
                xce, yce = (xk1 + xk2) / 2, (yk1 + yk2) / 2 # center point	
                # 使用宽度和高度比率调整检测框的坐标。
                fxk1 = (xk1 * width_ratio)	
                fyk1 = (yk1 * height_ratio)	
                fxk2 = (xk2 * width_ratio)	
                fyk2 = (yk2 * height_ratio)	
                fxce = (xce * width_ratio)	
                fyce = (yce * height_ratio)	
                # 将调整后的坐标转换为整数。
                xk1 = int(fxk1)	
                yk1 = int(fyk1)	
                xk2 = int(fxk2)	
                yk2 = int(fyk2)	
                xce = int(fxce)	
                yce = int(fyce)	
                xk1 = min(xk1, key_heatmaps.shape[3] - 1)
                yk1 = min(yk1, key_heatmaps.shape[2] - 1)
                xk2 = min(xk2, key_heatmaps.shape[3] - 1)
                yk2 = min(yk2, key_heatmaps.shape[2] - 1)
                xce = min(xce, key_heatmaps.shape[3] - 1)
                yce = min(yce, key_heatmaps.shape[2] - 1)
                # 如果使用高斯 bump，则通过调用 draw_gaussian 函数来绘制中心热图和关键热图。否则，直接在热图上设置值。
                if gaussian_bump:	
                    width  = detection[2] - detection[0]	
                    height = detection[3] - detection[1]	

                    width  = math.ceil(width * width_ratio)	
                    height = math.ceil(height * height_ratio)	

                    if gaussian_rad == -1:	
                        radius = gaussian_radius((height, width), gaussian_iou)	
                        radius = max(0, int(radius))	
                    else:	
                        radius = gaussian_rad	

                    #draw_gaussian(center_heatmaps[b_ind, int(category)], [xce, yce], radius)	
                    if 0 <= b_ind < batch_size and 0 <= int(category) < 10:
                        draw_gaussian(center_heatmaps[b_ind, int(category)], [xce, yce], radius)
                    else:
                        print(f"Invalid indices: b_ind={b_ind}, category={int(category)}")
                    draw_gaussian(key_heatmaps[b_ind, int(category)], [xk1, yk1], radius)	
                    draw_gaussian(key_heatmaps[b_ind, int(category)], [xk2, yk2], radius)	
                else:	
                    center_heatmaps[b_ind, int(category), yce, xce] = 1	
                    key_heatmaps[b_ind, int(category), yk1, xk1] = 1	
                    key_heatmaps[b_ind, int(category), yk2, xk2] = 1	
                # 为回归任务计算关键点和中心点的偏移。
                tag_ind = tag_lens_keys[b_ind]	
                #print(f"b_ind: {b_ind}")
                #print(f"tag_ind: {tag_ind}")
                key_regrs[b_ind, tag_ind, :] = [fxk1 - xk1, fyk1 - yk1]	
                key_regrs[b_ind, tag_ind+1, :] = [fxk2 - xk2, fyk2 - yk2]	
                center_regrs[b_ind, tag_ind//2, :] = [fxce - xce, fyce - yce]	
                # 计算关键标签和中心标签。
                key_tags[b_ind, tag_ind] = yk1 * output_size[1] + xk1	
                key_tags[b_ind, tag_ind+1] = yk2 * output_size[1] + xk2	
                center_tags[b_ind, tag_ind//2] = yce * output_size[1] + xce	

                # group target	
                keys_tag_len = tag_lens_keys[b_ind]	
                cens_tag_len = keys_tag_len // 2	
                group_target[b_ind, cens_tag_len, keys_tag_len: keys_tag_len + 2] = 1	
                # 更新标签长度，并检查是否超出最大长度。
                tag_lens_keys[b_ind] += 2	
                if tag_lens_keys[b_ind] >= max_tag_len-3:	
                    break	

                # 生成掩码，设置关键掩码和中心掩码，并记录中心标签长度。
                for b_ind in range(batch_size):	
                    tag_len = tag_lens_keys[b_ind]
                    key_masks[b_ind, :tag_len] = 1	
                    center_masks[b_ind, :tag_len//2] = 1	
                    tag_lens_cens[b_ind] = tag_len//2	

    #print(f"key_regrs: {key_regrs}")
    #print(f"center_regrs: {center_regrs}")
    #print(f"key_tags: {key_tags}")
    #print(f"center_tages: {center_tags}")
    #print(f"tag_lens_cens: {tag_lens_cens}")
    #print(f"tag_lens_keys: {tag_lens_keys}")
    images          = torch.from_numpy(images)
    key_heatmaps    = torch.from_numpy(key_heatmaps)
    center_heatmaps = torch.from_numpy(center_heatmaps)
    key_regrs       = torch.from_numpy(key_regrs)
    center_regrs    = torch.from_numpy(center_regrs)
    key_tags        = torch.from_numpy(key_tags)
    center_tags     = torch.from_numpy(center_tags)
    key_masks       = torch.from_numpy(key_masks)
    center_masks    = torch.from_numpy(center_masks)
    group_target    = torch.from_numpy(group_target)
    tag_lens_cens   = torch.from_numpy(tag_lens_cens)
    tag_lens_keys   = torch.from_numpy(tag_lens_keys)
    # xs 通常用来表示输入数据，而 ys 用来表示相应的标签或目标数据。
    return {
        "xs": [images, key_tags, center_tags, tag_lens_keys, tag_lens_cens],
        "ys": [key_heatmaps, center_heatmaps, key_masks, center_masks, key_regrs, center_regrs, group_target, tag_lens_cens, tag_lens_keys]
    }, k_ind

# globals() 是一个内置函数，返回一个代表当前全局符号表的字典。这个符号表始终针对当前模块（对函数或方法来说，是定义它们的模块），不是调用它的模块。
# 动态函数调用：globals()[system_configs.sampling_function] 这一部分是在从全局符号表中查找一个与 system_configs.sampling_function 字符串匹配的函数，并将其返回。
# 函数调用：一旦找到了匹配的函数，它就会像普通函数一样被调用，并传入 db, k_ind, debug 参数。
def sample_data(db, k_ind, debug=False):
    return globals()[system_configs.sampling_function](db, k_ind, debug)