from PIL import Image, ImageEnhance
import os
import time
import json
import cv2
import math
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials

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
            table_str += str(item) + ' '
    return table_str

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

def is_decimal(s):
    try:
        float(s)  # Try converting the string to a float
        return True
    except ValueError:
        return False  # Return False if the conversion fails

def data_range_estimation(Right, Bottom, OCR_results):
    # 提取符合条件的候选项

    for r in OCR_results:
        r['text'] = r['text'].replace(",", "")
    #print(f"OCR_results :{OCR_results}")
    #print(f"Right: {Right}, Bottom: {Bottom}")
    candidates = [r for r in OCR_results if r['bbox'][0] + r['bbox'][2] < Right - 4 and is_decimal(r['text'])]
    #print(f"candidates: {candidates}")
    # 找到最靠近 (Left, Bottom) 的候选项作为 rmin
    filtered_candidates = [r for r in candidates if float(r['text']) != 0]
    rmin = min(candidates, key=lambda r: abs(r['bbox'][0]) + abs(r['bbox'][1] - Bottom))

    # 找到最靠近 (Left, Top) 的候选项作为 rmax
    rmax = min(filtered_candidates, key=lambda r: abs(r['bbox'][0]) + abs(r['bbox'][1]))

    # 将文本转换为数字
    rmin['num'] = float(rmin['text'])
    rmax['num'] = float(rmax['text'])
    return rmax, rmin

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
    #print(f"max_val: {max_val}, min_val: {min_val}")
    # 计算 y 轴上的比例因子
    y_scale = (max_val - min_val) / (min_val_y - max_val_y)
    #print(max_val, min_val)
    #print(max_val_bbox, min_val_bbox)
    # 根据当前 val_bbox 离 min_val_bbox 的差距计算出当前值（val）
    val = min_val + (min_val_y - val_y) * y_scale

    return val

def filter_no_number(OCR_results):

    # 使用列表推导式来过滤掉 text 包含数字的项
    no_number_textss = [r for r in OCR_results if not any(char.isdigit() for char in r['text'])]
    
    return no_number_textss

def delete_y_label(min_val_tmp, texts):
    # 计算min_val_tmp的右上角和右下角坐标
    x_min, y_min, w, h = min_val_tmp
    top_right_x = x_min + w
    top_right_y = y_min
    bottom_right_x = x_min + w
    bottom_right_y = y_min + h

    # 新建一个列表来保存需要保留的矩形框
    filtered_texts = []

    for bbox in texts:
        x, y, w, h = bbox['bbox']
        # 计算当前矩形框的右上角和右下角坐标
        bbox_top_right_x = x + w
        bbox_top_right_y = y
        bbox_bottom_right_x = x + w
        bbox_bottom_right_y = y + h

        # 检查坐标差异是否在规定范围内
        if not (abs(top_right_x - bbox_top_right_x) <= 5 and abs(top_right_y - bbox_top_right_y) <= 5 and
                abs(bottom_right_x - bbox_bottom_right_x) <= 5 and abs(bottom_right_y - bbox_bottom_right_y) <= 5):
            # 如果不在规定范围内，则保留这个矩形框
            filtered_texts.append(bbox)

    return filtered_texts
def convert_to_xywh(bounding_box):
    # 提取所有x和y坐标
    x_coords = bounding_box[0::2]
    y_coords = bounding_box[1::2]

    # 计算最小矩形的左上角坐标
    x_min = min(x_coords)
    y_min = min(y_coords)

    # 计算宽度和高度
    w = max(x_coords) - x_min
    h = max(y_coords) - y_min

    return x_min, y_min, w, h

def ocr_with_azure():
    subscription_key = os.environ["VISION_KEY"]
    endpoint = os.environ["VISION_ENDPOINT"]

    computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

    image = Image.open("./test_img.png")
    enh_con = ImageEnhance.Contrast(image)
    contrast = 2.0
    image = enh_con.enhance(contrast)
    # image = image.convert('L')
    # image = image.resize((800, 800))
    image.save('OCR_temp.png')
    image = open("./OCR_temp.png", "rb")

    read_response = computervision_client.read_in_stream(image,  raw=True)

    read_operation_location = read_response.headers["Operation-Location"]
    operation_id = read_operation_location.split("/")[-1]

    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)

    tesseract_format = {
        'left': [],
        'top': [],
        'width': [],
        'height': [],
        'text': []
    }
    image = cv2.imread("./test_img.png")
    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results:
            for line in text_result.lines:
                # Populate the dictionary
                tesseract_format['text'].append(line.text)
                x, y, w, h = convert_to_xywh(line.bounding_box)
                print(line.bounding_box)
                tesseract_format['left'].append(x)
                tesseract_format['top'].append(y)
                tesseract_format['width'].append(w)
                tesseract_format['height'].append(h)
    return tesseract_format

def find_topmost_bbox(ocr_results):
    if not ocr_results:
        return None

    # Sort the OCR results by the y coordinate of the top-left corner
    sorted_results = sorted(ocr_results, key=lambda r: r['bbox'][1])

    # The first element after sorting will be the topmost
    topmost_bbox = sorted_results[0]

    return topmost_bbox

def find_bottommost_bbox(ocr_results):
    if not ocr_results:
        return None

    # Sort by the y coordinate of the bottom-right corner
    sorted_results = sorted(ocr_results, key=lambda r: r['bbox'][3], reverse=True)

    # The first element is the bottommost
    bottommost_bbox = sorted_results[0]

    return bottommost_bbox

def find_leftmost_bbox(ocr_results):
    if not ocr_results:
        return None

    # Sort by the x coordinate of the top-left corner
    sorted_results = sorted(ocr_results, key=lambda r: r['bbox'][0])

    # The first element is the leftmost
    leftmost_bbox = sorted_results[0]

    return leftmost_bbox

def Ocr(chart_type):
    image = cv2.imread("./test_img.png")
    height, width, _ = image.shape

    # 计算坐标
    top = 0
    left = 0
    bottom = height - 1
    right = width - 1
    json_result = {"results":[]}
    result = ocr_with_azure()
    for i in range(len(result['text'])):
        #if int(result['conf'][i]) > 0:
        text = result['text'][i]
        if(text.strip() != ''):
            bbox_info=(result['left'][i], result['top'][i], result['width'][i], result['height'][i])
            json_result['results'].append({"text":text, "bbox": bbox_info})
            print(f"文字: {result['text'][i]}, 边界框: ({result['left'][i]}, {result['top'][i]}, {result['width'][i]}, {result['height'][i]})")
            x, y, w, h = result['left'][i], result['top'][i], result['width'][i], result['height'][i]
            #print(x, y, w, h)
            # 使用OpenCV绘制矩形
            cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
    cv2.imwrite('output.png', image)
    image = cv2.imread("./test_img.png")
    with open(os.getcwd() + "/evaluation/KPGroupingbest_full.json", "r") as f:
        annotations = json.load(f)
    xs = []
    ys = []
    max_val = -1.0
    min_val = -1.0
    max_val_bbox = [-1.0, -1.0, -1.0, -1.0]
    min_val_bbox = [-1.0, -1.0, -1.0, -1.0]
    no_number_texts = filter_no_number(json_result['results'])
    texts = json_result['results']
    if(chart_type != 'pie'):
        y_axis_title = find_leftmost_bbox(no_number_texts)['text']
        x_axis_title = find_bottommost_bbox(no_number_texts)['text']
    else:
        y_axis_title = "None"
        x_axis_title = "None"

    chart_title = find_topmost_bbox(no_number_texts)['text']
    #print(json_result['results'])
    sorted_groups = sorted(annotations['test_img.png'][2], key=lambda group: group[0])
    # 收集要删除的元素索引
    indexes_to_remove = set()

    for i in range(len(sorted_groups) - 1):
        current_group = sorted_groups[i]
        next_group = sorted_groups[i + 1]
    
        # 检查第一位和第二位的差值是否都小于1
        if abs(current_group[0] - next_group[0]) < 5 and abs(current_group[1] - next_group[1]) < 5:
            # 根据第二位的大小决定要删除的元素
            if current_group[1] < next_group[1]:
                indexes_to_remove.add(i)
            else:
                indexes_to_remove.add(i + 1)

    # 反向删除标记的元素
    for index in sorted(indexes_to_remove, reverse=True):
        del sorted_groups[index]
    for group in sorted_groups:
        category = group[-1]
        #print(category)
        if(category == 0):
            if(chart_type != "vbar_categorical"):
                continue
            if(max_val == -1):
                max_val_tmp, min_val_tmp = data_range_estimation(right, bottom, texts)
                max_val = max_val_tmp['num']
                min_val = min_val_tmp['num']
                max_val_bbox = max_val_tmp['bbox']
                min_val_bbox = min_val_tmp['bbox']
                print(f"max_val: {max_val_tmp}")
                print(f"min_val: {min_val_tmp}")
                texts = delete_y_label(min_val_bbox, texts)
                texts = delete_y_label(max_val_bbox, texts)
            if(group[2] > group[4] and group[3] > group[5]):
                group[2], group[4] = group[4], group[2]
                group[3], group[5] = group[5], group[3]
            xs.append(calculate_bar_val(max_val, max_val_bbox, min_val, min_val_bbox, group[2:6]))
            print(f"group: {group}")
            ys.append(find_closest_text(group[0:2], texts))
        elif(category == 1):
            if(chart_type != 'line'):
                continue
            if(max_val == -1):
                max_val_tmp, min_val_tmp = data_range_estimation(right, bottom, texts)
                max_val = max_val_tmp['num']
                min_val = min_val_tmp['num']
                max_val_bbox = max_val_tmp['bbox']
                min_val_bbox = min_val_tmp['bbox']
                print(f"max_val: {max_val_tmp}")
                print(f"min_val: {min_val_tmp}")
                texts = delete_y_label(min_val_bbox, texts)
                texts = delete_y_label(max_val_bbox, texts)
                print(f"texts: {texts}")
            for k in range(1, len(group[:-1])//2):
                key_in_group = group[2*k:2*k+2]
                xs.append(calculate_bar_val(max_val, max_val_bbox, min_val, min_val_bbox, key_in_group))
                ys.append(find_closest_text(key_in_group, texts))
            continue
        elif(category == 2):
            if(chart_type != 'pie'):
                continue
            xs.append(sector_area_ratio(group[0:6]))
            ys.append(find_closest_text([group[0],group[1]], no_number_texts))
            continue
    converted_table =  convert_to_table_format(ys) + "& "
    converted_table += convert_to_table_format(xs)
    converted_table = converted_table + "Chart Type: " + chart_type + " "
    converted_table = converted_table + "Title: " + chart_title + " "
    converted_table = converted_table + "x_axis_title: " + x_axis_title + " "
    converted_table = converted_table + "y_axis_title: " + y_axis_title + " "
    return converted_table