import cv2
import numpy as np
import torch
import os
import math
from config import system_configs
from .sampling_utils import _full_image_crop, _resize_image, _clip_detections, draw_gaussian, gaussian_radius
from img_utils import normalize_
import matplotlib.pyplot as plt
import os
from img_utils import color_jittering_, lighting_

def save_key_heatmaps(key_heatmaps, save_dir='heatmaps'):
    """
    Save the key heatmaps for each category as images.

    Parameters:
    - key_heatmaps: 4D NumPy array of shape (batch_size, categories, output_size[0], output_size[1])
    - save_dir: Directory where to save the heatmap images.
    """

    # 创建保存热图的目录，如果不存在的话
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    batch_size, categories, _, _ = key_heatmaps.shape

    for b in range(batch_size):
        for c in range(categories):
            # 提取单个热图
            heatmap = key_heatmaps[b, c, :, :]

            plt.imshow(heatmap, cmap='hot', interpolation='nearest')
            plt.colorbar()

            plt.title(f'Batch {b + 1}, Category {c + 1} Key Heatmap')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')

            # 保存热图为图片
            plt.savefig(os.path.join(save_dir, f'{b+1}_category_{c+1}_key_heatmap.png'))

            # 清除当前图形，以便绘制下一个
            plt.clf()

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


def kp_sampling(db, k_ind):
    batch_size = system_configs.batch_size
    data_rng = system_configs.data_rng
    categories   = db.configs["categories"]
    input_size   = db.configs["input_size"]
    output_size  = db.configs["output_sizes"][0]
    gaussian_bump = db.configs["gaussian_bump"]
    gaussian_iou  = db.configs["gaussian_iou"]
    gaussian_rad  = db.configs["gaussian_radius"]
    rand_color = db.configs["rand_color"]
    lighting = db.configs["lighting"]
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
        if k_ind == 0:
            db.shuffle_inds()
        flag = False
        while not flag:
            db_ind = db.db_inds[k_ind]
            k_ind = (k_ind + 1) % db_size
            # reading image 
            image_file = db.image_file(db_ind)
            if(os.path.exists(image_file) and len(db.detections(db_ind)) <= max_tag_len//3) and len(db.detections(db_ind)) > 0: 
                #print(db.detections(db_ind))
                (detections, categories) = db.detections(db_ind)
                if(len(categories)):
                    flag = True
        image = cv2.imread(image_file)
        ori_size = image.shape
            #print(temp)
        #print(f"k_ind: {k_ind}")
        (detections, categories) = db.detections(db_ind)
        #print(detections)
        #print(f"Detections: {detections}")
        #print(f"Length of detection: {len(detections)}")
        #print(f"Categories: {categories}")
        detections = detections.tolist()
        max_len = 0
        cur_group_len = 0
        for i in range(len(detections)):
            if(categories[i] == 3):
                detection = detections[i]
                if len(detection) < 5:
                    print("Insufficient elements in the detection list.")
                    print(len(detection))
                    print(image_file)
                    continue
                xce, yce = get_center((detection[0], detection[1]), (detection[2], detection[3]), (detection[4], detection[5]))
                detections[i] = np.concatenate((detection[:6], [xce, yce], [detection[-1]]), axis=0)
            elif(categories[i] == 2):
                cur_group_len += 1
                if(cur_group_len > max_group_len):
                    del detections[i]
                    del categories[i]
            max_len = max(max_len, len(detections[i]))
        for i in range(len(detections)):
            if len(detections[i]) < max_len: detections[i] = np.pad(detections[i], (0, max_len - len(detections[i])), 'constant', constant_values=(0, 0)) 
        #print(detections)
        detections = np.array(detections)
        # cropping an image randomly
        image, detections = _full_image_crop(image, detections)
        scale = 1
        #cv2.imwrite('cropped.png', image)
        #print(f"Cropped detections: {detections}")
        image, detections = _resize_image(image, detections, input_size)
        #cv2.imwrite('resized.png', image)
        #print(f"Resized detections: {detections}")
        detections = _clip_detections(image, detections)
        width_ratio  = output_size[1] / input_size[1]
        height_ratio = output_size[0] / input_size[0]
        #print(f"input size:{input_size}")
        #print(f"width ratio: {width_ratio}, height ratio: {height_ratio}")
        #print(f"Clipped detections: {detections}")
        #将图像数组的数据类型转换为浮点型（float32）。在 NumPy 中，astype 方法用于更改数组的数据类型。
        image = image.astype(np.float32) / 255.
        if rand_color:
            color_jittering_(data_rng, image)
            if lighting:
                lighting_(data_rng, image, 0.1, db.eig_val, db.eig_vec)
        normalize_(image, db.mean, db.std)
        images[b_ind] = image.transpose((2, 0, 1))
        for ind, (detection, _category) in enumerate(zip(detections, categories)):
            #print(f"ind: {ind}, detection: {detection}, category: {category}")
            category = int(_category)
            if(category == 2):
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
                #xce = min(xce, key_heatmaps.shape[3] - 1)
                #yce = min(yce, key_heatmaps.shape[2] - 1)
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
                            key_heatmaps[b_ind, category, detection[2 * k + 1],detection[2 * k]] = 1
                            center_heatmaps[b_ind, category, yce, xce] = 1

                for k in range(int(len(detection) / 2)):
                    if not bad_p(detection[2*k], detection[2*k+1], output_size):
                        #print(f"tag_lens_keys[{b_ind}]: {tag_lens_keys[b_ind]}")
                        if tag_lens_keys[b_ind] >= max_tag_len - 1:
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
            elif(category == 3):
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
                    center_heatmaps[b_ind, category, yce, xce] = 1
                    key_heatmaps[b_ind, category, yk1, xk1] = 1
                    key_heatmaps[b_ind, category, yk2, xk2] = 1
                    key_heatmaps[b_ind, category, yk3, xk3] = 1

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
                #print(f"bind:{b_ind}")
                #print(f"category:{category}")
                xk1, yk1 = detection[0], detection[1] # top left point	
                xk2, yk2 = detection[2], detection[3] # bottom right point	
                #print(xk1, yk1)
                #print(xk2, yk2)
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
                #print(xk1, yk1)
                #print(xk2, yk2)
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
                    center_heatmaps[b_ind, category, yce, xce] = 1	
                    key_heatmaps[b_ind, category, yk1, xk1] = 1
                    key_heatmaps[b_ind, category, yk2, xk2] = 1
                #print(xk1, yk1)
                #print(xk2, yk2)
                #print(yce, xce)
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
                if tag_lens_keys[b_ind] >= max_tag_len-2:	
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
    #save_key_heatmaps(center_heatmaps)
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
# 函数调用：一旦找到了匹配的函数，它就会像普通函数一样被调用，并传入 db, k_ind 参数。
def sample_data(db, k_ind):
    return globals()[system_configs.sampling_function](db, k_ind)