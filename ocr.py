from PIL import Image, ImageEnhance
import pytesseract
import os
import logging
from pytesseract import Output
import cv2
import json
import math

def distance(x1, y1, x2, y2):
    """Calculate the distance between two points (x1, y1) and (x2, y2)."""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def angle_between_points(x1, y1, x2, y2):
    """Calculate the angle between two points (x1, y1) and (x2, y2) with respect to the origin."""
    return math.atan2(y2 - y1, x2 - x1)

def sector_area_ratio(points):
    """
    Calculate the area ratio of a sector given three critical points:
    [center_x, center_y, edge_1_x, edge_1_y, edge_2_x, edge_2_y]
    The area ratio is the area of the sector divided by the area of the circle.
    """
    edge_1_x, edge_1_y, edge_2_x, edge_2_y, center_x, center_y = points
    
    # Step 1: Calculate the radius of the sector
    radius = distance(center_x, center_y, edge_1_x, edge_1_y)
    
    # Step 2: Calculate angles for the two edge points
    angle1 = angle_between_points(center_x, center_y, edge_1_x, edge_1_y)
    angle2 = angle_between_points(center_x, center_y, edge_2_x, edge_2_y)
    
    # Calculate the angle between the two radii of the sector
    theta = abs(angle2 - angle1)
    
    # Make sure theta is between 0 and 2*pi
    if theta > 2 * math.pi:
        theta = 2 * math.pi - theta
    
    # Step 3: Calculate the area of the sector
    sector_area = 0.5 * theta * radius ** 2
    
    # Calculate the area of the circle
    circle_area = math.pi * radius ** 2
    
    # Calculate the area ratio
    area_ratio = sector_area / circle_area
    
    return area_ratio

def sector_area(points):
    """
    Calculate the area of a sector given three critical points:
    [center_x, center_y, edge_1_x, edge_1_y, edge_2_x, edge_2_y]
    """
    center_x, center_y, edge_1_x, edge_1_y, edge_2_x, edge_2_y = points
    
    # Step 1: Calculate the radius of the sector
    radius = distance(center_x, center_y, edge_1_x, edge_1_y)
    
    # Step 2: Calculate angles for the two edge points
    angle1 = angle_between_points(center_x, center_y, edge_1_x, edge_1_y)
    angle2 = angle_between_points(center_x, center_y, edge_2_x, edge_2_y)
    
    # Calculate the angle between the two radii of the sector
    theta = abs(angle2 - angle1)
    
    # Make sure theta is between 0 and 2*pi
    if theta > 2 * math.pi:
        theta = 2 * math.pi - theta
    
    # Step 3: Calculate the area of the sector
    area = 0.5 * theta * radius ** 2
    
    return area

def ocr_with_tesseract(filename):
    """ Use Tesseract to extract text from image.
    :param filename: Your file path & name.
    :param language: Language code to be used in OCR.
                    List of available language codes can be found at https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html
                    Defaults to 'eng'.
    :return: Result in plain text format.
    """
    #image = Image.open(filename)
    #enh_con = ImageEnhance.Contrast(image)
    #contrast = 2.0
    #image = enh_con.enhance(contrast)
    #image.save('OCR_temp3.png')
    #raw_text = pytesseract.image_to_string(Image.open('OCR_temp.png'))
    # print(raw_text)
    #detected_language = detect(raw_text)
    #print(detected_language)
    detected_language = 'eng'
    #enh_con = ImageEnhance.Contrast(image)
    #contrast = 2.0
    #image = enh_con.enhance(contrast)
    #image.save('OCR_temp.png')
    #image = cv2.imread('OCR_temp.png')
    image = cv2.imread(filename)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化
    binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # 降噪
    denoised_image = cv2.fastNlMeansDenoising(binary_image, None, 30, 7, 21)
    cv2.imwrite('denoised.png', denoised_image)
    cv2.imwrite('binary.png', binary_image)
    cv2.imwrite('gray.png', gray_image)
    #scale_factor = 2
    #scaled_image = cv2.resize(denoised_image, (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor)))
    # image = Image.open('denoised.png')
    # enh_con = ImageEnhance.Contrast(image)
    # contrast = 2.0
    # image = enh_con.enhance(contrast)
    # image.save('enhanced.png')
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #dilated_image = cv2.dilate(scaled_image, kernel, iterations=1)
    #cv2.imwrite('dilated.png', dilated_image)
    #eroded_image = cv2.erode(dilated_image, kernel, iterations=1)
    #cv2.imwrite('eroded.png', eroded_image)
    result = pytesseract.image_to_data(denoised_image, lang=detected_language, config='--psm 12', output_type=Output.DICT)
    #os.remove('OCR_temp.png')
    return result
def filter_no_number(OCR_results):
    """
    输入：
    - OCR_results: 一个字典列表，其中每个字典都有 'text' 和 'bbox' 两个键

    输出：一个新的字典列表，其中不包含 text 包含数字的项
    """
    
    # 使用列表推导式来过滤掉 text 包含数字的项
    no_number_textss = [r for r in OCR_results if not any(char.isdigit() for char in r['text'])]
    
    return no_number_textss

def data_range_estimation(Right, Bottom, OCR_results):
    # 提取符合条件的候选项
    for r in OCR_results:
        r['text'] = r['text'].replace(",", "")
    candidates = [r for r in OCR_results if r['bbox'][0] + r['bbox'][2] < Right - 4 and r['text'].isdigit()]

    # 找到最靠近 (Left, Bottom) 的候选项作为 rmin
    filtered_candidates = [r for r in candidates if float(r['text']) != 0]
    rmin = min(candidates, key=lambda r: abs(r['bbox'][0]) + abs(r['bbox'][1] - Bottom))

    # 找到最靠近 (Left, Top) 的候选项作为 rmax
    rmax = min(filtered_candidates, key=lambda r: abs(r['bbox'][0]) + abs(r['bbox'][1]))

    # 将文本转换为数字
    rmin['num'] = float(rmin['text'])
    rmax['num'] = float(rmax['text'])

    # 计算Y轴的刻度
    #Yscale = (rmin['num'] - rmax['num']) / (rmax['bbox'][1] - rmin['bbox'][1])

    # 计算Y轴的最小值
    #Ymin = rmin['num'] - Yscale * ((Bottom - rmin['t'] + rmin['b']) / 2)

    ## 计算Y轴的最大值
    #Ymax = rmax['num'] + Yscale * ((rmax['t'] + rmax['b']) / 2 - Top)

    return rmax, rmin

def find_bounding_boxes(top_left_points, bottom_right_points, is_vertical=True, threshold=0.4):
    # γ 和 ν 权重的定义，取决于是垂直条还是水平条
    gamma = 1 if is_vertical else 0
    nu = 0 if is_vertical else 1
    
    bounding_boxes = []

    # 遍历每个顶部左侧点
    for ptl in top_left_points:
        # 过滤概率低于阈值的点
        if ptl['prob'] < threshold:
            continue
            
        # 找到底部右侧点，这些点在顶部左侧点的右侧（仅适用于垂直条形图）
        candidates = [pbr for pbr in bottom_right_points if pbr['prob'] >= threshold and (pbr['x'] > ptl['x'] or not is_vertical)]

        # 根据定义的距离度量找到最近的底部右侧点
        closest_pbr = min(candidates, key=lambda pbr: gamma * abs(pbr['x'] - ptl['x']) + nu * abs(pbr['y'] - ptl['y']))

        # 创建边界框并添加到列表中
        bounding_box = {
            'top_left': (ptl['x'], ptl['y']),
            'bottom_right': (closest_pbr['x'], closest_pbr['y'])
        }
        bounding_boxes.append(bounding_box)

    return bounding_boxes

#def try_math(image_path, cls_info):
    #title_list = [1, 2, 3]
    #title2string = {}
    #max_value = 1
    #min_value = 0
    #max_y = 0
    #min_y = 1
    #word_infos = ocr_result(image_path)
    #for id in title_list:
        #if id in cls_info.keys():
            #predicted_box = cls_info[id]
            #words = []
            #for word_info in word_infos:
                #word_bbox = [word_info["boundingBox"][0], word_info["boundingBox"][1], word_info["boundingBox"][4], word_info["boundingBox"][5]]
                #if check_intersection(predicted_box, word_bbox) > 0.5:
                    #words.append([word_info["text"], word_bbox[0], word_bbox[1]])
            #words.sort(key=lambda x: x[1]+10*x[2])
            #word_string = ""
            #for word in words:
                #word_string = word_string + word[0] + ' '
            #title2string[id] = word_string
    #if 5 in cls_info.keys():
        #plot_area = cls_info[5]
        #y_max = plot_area[1]
        #y_min = plot_area[3]
        #x_board = plot_area[0]
        #dis_max = 10000000000000000
        #dis_min = 10000000000000000
        #for word_info in word_infos:
            #word_bbox = [word_info["boundingBox"][0], word_info["boundingBox"][1], word_info["boundingBox"][4], word_info["boundingBox"][5]]
            #word_text = word_info["text"]
            #word_text = re.sub('[^-+0123456789.]', '',  word_text)
            #word_text_num = re.sub('[^0123456789]', '', word_text)
            #word_text_pure = re.sub('[^0123456789.]', '', word_text)
            #if len(word_text_num) > 0 and word_bbox[2] <= x_board+4:
                #dis2max = math.sqrt(math.pow((word_bbox[0]+word_bbox[2])/2-x_board, 2)+math.pow((word_bbox[1]+word_bbox[3])/2-y_max, 2))
                #dis2min = math.sqrt(math.pow((word_bbox[0] + word_bbox[2]) / 2 - x_board, 2) + math.pow(
                    #(word_bbox[1] + word_bbox[3]) / 2 - y_min, 2))
                #y_mid = (word_bbox[1]+word_bbox[3])/2
                #if dis2max <= dis_max:
                    #dis_max = dis2max
                    #max_y = y_mid
                    #max_value = float(word_text_pure)
                    #if word_text[0] == '-':
                        #max_value = -max_value
                #if dis2min <= dis_min:
                    #dis_min = dis2min
                    #min_y = y_mid
                    #min_value = float(word_text_pure)
                    #if word_text[0] == '-':
                        #min_value = -min_value
        #delta_min_max = max_value-min_value
        #delta_mark = min_y - max_y
        #delta_plot_y = y_min - y_max
        #delta = delta_min_max/delta_mark
        #if abs(min_y-y_min)/delta_plot_y > 0.1:
            #print(abs(min_y-y_min)/delta_plot_y)
            #print("Predict the lower bar")
            #min_value = int(min_value + (min_y-y_min)*delta)

    #return title2string, round(min_value, 2), round(max_value, 2)

def find_closest_text(input_bbox, OCR_results):
    """
    输入：
    - input_bbox: 输入的边界框，形如 (x, y, width, height)
    - OCR_results: 一个字典列表，其中每个字典都有 'bbox' 和 'text' 两个键

    输出：与输入 bbox 最接近的 'text'
    """
    
    # 如果 OCR_results 为空，返回 None
    if not OCR_results:
        return None

    # 计算两个 bbox 之间的距离（我们使用左上角坐标的欧几里得距离）
    def distance(bbox1, bbox2):
        return ((bbox1[0] - bbox2[0]) ** 2 + (bbox1[1] - bbox2[1]) ** 2) ** 0.5

    # 找到与输入 bbox 最接近的 OCR_result
    closest_result = min(OCR_results, key=lambda r: distance(input_bbox, r['bbox']))
    OCR_results.remove(closest_result)
    return closest_result['text']

def print_items(xs):
    for i, item in enumerate(xs):
        if i < len(xs) - 1:
            print(item, end=' | ')
        else:
            print(item, end=' &\n')

def convert_to_table_format(xs):
    """
    Convert a list of items to a table format string.
    Each item is separated by ' | ' except the last one, which is followed by ' &\n'.
    """
    table_str = ""
    for i, item in enumerate(xs):
        if i < len(xs) - 1:
            table_str += str(item) + ' | '
        else:
            table_str += str(item) + ' &\n'
    return table_str

def calculate_bar_val(max_val, max_val_bbox, min_val, min_val_bbox, val_bbox):
    """
    输入：
    - max_val: y轴上的最大值
    - max_val_bbox: 最大值的边界框坐标
    - min_val: y轴上的最小值
    - min_val_bbox: 最小值的边界框坐标
    - val_bbox: 当前值的边界框坐标

    输出：根据当前 val_bbox 计算出的当前值（val）
    """

    # 获取 y 轴坐标
    max_val_y = max_val_bbox[1]
    min_val_y = min_val_bbox[1]
    val_y = val_bbox[1]

    # 计算 y 轴上的比例因子
    y_scale = (max_val - min_val) / (min_val_y - max_val_y)
    #print(max_val, min_val)
    #print(max_val_bbox, min_val_bbox)
    # 根据当前 val_bbox 离 min_val_bbox 的差距计算出当前值（val）
    val = min_val + (min_val_y - val_y) * y_scale

    return val

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    image_file_path = "./img/test_img"
    # 读取图像
    image = cv2.imread(image_file_path)

    # 获取图像的高度和宽度
    height, width, _ = image.shape

    # 计算坐标
    top = 0
    left = 0
    bottom = height - 1
    right = width - 1
    json_result = {"results":[]}
    result = ocr_with_tesseract(image_file_path)
    for i in range(len(result['text'])):
        #if int(result['conf'][i]) > 0:
        text = result['text'][i]
        if(text.strip() != ''):
            bbox_info=(result['left'][i], result['top'][i], result['width'][i], result['height'][i])
            json_result['results'].append({"text":text, "bbox": bbox_info})
            print(f"文字: {result['text'][i]}, 边界框: ({result['left'][i]}, {result['top'][i]}, {result['width'][i]}, {result['height'][i]}), 置信度:{result['conf'][i]}")
    image = cv2.imread(image_file_path)
    cv2.imwrite('output.png', image)
    with open(os.getcwd() + "/evaluation/KPGrouping50000.json", "r") as f:
        annotations = json.load(f)
    y_axis_title = 'None'
    x_axis_title = 'None'
    chart_type = 'None'
    chart_title = 'None'
    xs = []
    ys = []
    max_val = -1.0
    min_val = -1.0
    val_scale = -1.0
    max_val_bbox = [-1.0, -1.0, -1.0, -1.0]
    min_val_bbox = [-1.0, -1.0, -1.0, -1.0]
    no_number_texts = filter_no_number(json_result['results'])
    texts = json_result['results']
    for groups in annotations['test_img.png'][2]:
        category = groups
        if(category == '1'):
            if(chart_type != "Bar"):
                continue
            if(max_val == -1):
                max_val_tmp, min_val_tmp = data_range_estimation(right, bottom, json_result['results'])
                max_val = max_val_tmp['num']
                min_val = min_val_tmp['num']
                max_val_bbox = max_val_tmp['bbox']
                min_val_bbox = min_val_tmp['bbox']
            xs.append(calculate_bar_val(max_val, max_val_bbox, min_val, min_val_bbox, i['bbox']))
            ys.append(find_closest_text(i['bbox'], no_number_texts))
        elif(category == 2):
            if(chart_type != 'Line'):
                continue
            if(max_val == -1):
                max_val_tmp, min_val_tmp = data_range_estimation(right, bottom, json_result['results'])
                max_val = max_val_tmp['num']
                min_val = min_val_tmp['num']
                max_val_bbox = max_val_tmp['bbox']
                min_val_bbox = min_val_tmp['bbox']
            j = 0
            while(j < len(i['bbox'])):
                cur_bbox = [i['bbox'][j], i['bbox'][j + 1]]
                xs.append(calculate_bar_val(max_val, max_val_bbox, min_val, min_val_bbox, cur_bbox))
                ys.append(find_closest_text(cur_bbox, texts))
                j = j + 2
            continue
        elif(category == 3):
            if(chart_type != 'Pie'):
                continue
            xs.append(sector_area_ratio(i['bbox']))
            ys.append(find_closest_text([(i['bbox'][0] + i['bbox'][2] + i['bbox'][4])/3,(i['bbox'][1] + i['bbox'][3] + i['bbox'][5])/3], no_number_texts))
            continue
        elif(category == '5'):
            y_axis_title = find_closest_text(i['bbox'], no_number_texts)
        elif(category == '6'):
            chart_title = find_closest_text(i['bbox'],no_number_texts)
        elif(category == '7'):
            x_axis_title = find_closest_text(i['bbox'],no_number_texts)
    converted_table = convert_to_table_format(xs)
    converted_table += convert_to_table_format(ys)
    converted_table = converted_table + "Chart Type: " + chart_type + " &"
    converted_table = converted_table + "Title: " + chart_title + " &"
    converted_table = converted_table + "X Axis Title: " + x_axis_title + " &"
    converted_table = converted_table + "Y Axis Title: " + y_axis_title + " &"