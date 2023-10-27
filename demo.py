#!/root/miniconda3/envs/ChartLLM/bin/python
import gradio as gr
import logging
import os
from transformers import(
    AutoTokenizer,
    T5ForConditionalGeneration
)
from PIL import Image, ImageDraw
import subprocess
import json
from pytesseract import Output
import pytesseract
import math
import cv2

# 设置命令和参数
extraction_command = "val_extraction.py"
extraction_params = [
    "--img_path", "./img",
    "--save_path", "evaluation",
    "--model_type", "KPGrouping",
    "--cache_path", "./data/cache/",
    "--data_dir", "./",
    "--iter", "50000"
]

# 合并命令和参数
t5_tokenizer = AutoTokenizer.from_pretrained("./data/cache/t5_output/checkpoint-70000")
t5_model = T5ForConditionalGeneration.from_pretrained("./data/cache/t5_output/checkpoint-70000")

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

def Ocr(chart_type):
    image = cv2.imread("./img/test_img.png")
    height, width, _ = image.shape

    # 计算坐标
    top = 0
    left = 0
    bottom = height - 1
    right = width - 1
    json_result = {"results":[]}
    result = ocr_with_tesseract("./img/test_img.png")
    for i in range(len(result['text'])):
        #if int(result['conf'][i]) > 0:
        text = result['text'][i]
        if(text.strip() != ''):
            bbox_info=(result['left'][i], result['top'][i], result['width'][i], result['height'][i])
            json_result['results'].append({"text":text, "bbox": bbox_info})
            print(f"文字: {result['text'][i]}, 边界框: ({result['left'][i]}, {result['top'][i]}, {result['width'][i]}, {result['height'][i]}), 置信度:{result['conf'][i]}")
    image = cv2.imread("./img/test_img.png")
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
    chart_type = "Pie"
    for group in annotations['test_img.png'][2]:
        category = group[-1]
        if(category == 1):
            if(chart_type != "Bar"):
                continue
            if(max_val == -1):
                max_val_tmp, min_val_tmp = data_range_estimation(right, bottom, json_result['results'])
                max_val = max_val_tmp['num']
                min_val = min_val_tmp['num']
                max_val_bbox = max_val_tmp['bbox']
                min_val_bbox = min_val_tmp['bbox']
            xs.append(calculate_bar_val(max_val, max_val_bbox, min_val, min_val_bbox, group[2:6]))
            ys.append(find_closest_text(group[2:6], no_number_texts))
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
            while(j < len(group[2:6])):
                cur_bbox = [group[2:6][j], group[2:6][j + 1]]
                xs.append(calculate_bar_val(max_val, max_val_bbox, min_val, min_val_bbox, cur_bbox))
                ys.append(find_closest_text(cur_bbox, texts))
                j = j + 2
            continue
        elif(category == 3):
            if(chart_type != 'Pie'):
                continue
            xs.append(sector_area_ratio(group[0:6]))
            ys.append(find_closest_text([group[0],group[1]], no_number_texts))
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
    return converted_table


def Visualization(im):
    file_name = os.getcwd() + "/evaluation/KPGrouping50000.json"
    with open(file_name) as f:
        data = json.load(f)
    thres = 0.4
    draw = ImageDraw.Draw(im)
    chart_type = None
    img_name = "test_img.png"
    groups = data[img_name][2]
    for group in groups:
        # 提取分组中的中心点坐标。
        cen_in_group = group[0:2]
        # 遍历分组中的关键点。
        for k in range(1, len(group[:-1])//2):
            key_in_group = group[2*k:2*k+2]
            draw.line([tuple(cen_in_group),tuple(key_in_group)], fill ="blue", width = 1)
    if(sum(1 for k in data[img_name][1]['1'] if k[0] > thres) > 4):
        chart_type = "Bar"
        keys = data[img_name][0]['1']
        cens = data[img_name][1]['1']
        for bbox in keys:
            if bbox[0] > thres:
                draw.ellipse((bbox[2]-5, bbox[3]-5, bbox[2]+5, bbox[3]+5), fill=(255, 0, 0), outline=(0, 0, 0))
        for bbox in cens:
            if bbox[0] > thres:
                draw.ellipse((bbox[2]-5, bbox[3]-5, bbox[2]+5, bbox[3]+5), fill=(0, 255, 0), outline=(0, 0, 0))
    elif(sum(1 for k in data[img_name][1]['3'] if k[0] > thres) > 3):
        chart_type = "Pie"
        keys = data[img_name][0]['3']
        cens = data[img_name][1]['3']
        for bbox in keys:
            if bbox[0] > thres:
                draw.ellipse((bbox[2]-5, bbox[3]-5, bbox[2]+5, bbox[3]+5), fill=(255, 0, 0), outline=(0, 0, 0))
        for bbox in cens:
            if bbox[0] > thres:
                draw.ellipse((bbox[2]-5, bbox[3]-5, bbox[2]+5, bbox[3]+5), fill=(0, 255, 0), outline=(0, 0, 0))
    else:
        chart_type = "Line"
        keys = data[img_name][0]['2']
        cens = data[img_name][1]['2']
        for bbox in keys:
            if bbox[0] > thres:
                draw.ellipse((bbox[2]-5, bbox[3]-5, bbox[2]+5, bbox[3]+5), fill=(255, 0, 0), outline=(0, 0, 0))
        for bbox in cens:
            if bbox[0] > thres:
                draw.ellipse((bbox[2]-5, bbox[3]-5, bbox[2]+5, bbox[3]+5), fill=(0, 255, 0), outline=(0, 0, 0))
    thres = 0.
    for category in ['4', '5', '6', '7', '8']:
        keys = data[img_name][0][category]
        cens = data[img_name][1][category]
        for bbox in keys:
            if bbox[0] > thres:
                draw.ellipse((bbox[2]-5, bbox[3]-5, bbox[2]+5, bbox[3]+5), fill=(255, 0, 0), outline=(0, 0, 0))
        for bbox in cens:
            if bbox[0] > thres:
                draw.ellipse((bbox[2]-5, bbox[3]-5, bbox[2]+5, bbox[3]+5), fill=(0, 255, 0), outline=(0, 0, 0))
    return im, chart_type

def ChartQA(chart_img, question, model):
    #print("Current Working Directory:", os.getcwd())
    chart_img.save("./img/test_img.png")
    subprocess.run(["python3", os.getcwd() + "/" + extraction_command] + extraction_params)
    visualized_img, chart_type = Visualization(chart_img)
    chart_table = Ocr(chart_type)
    answer = "ERROR"
    if(model == "T5"):
        question = "Question: " + question
        chart_info = "Table: " + chart_table
        query = question + chart_info
        #print(query)
        input_ids = t5_tokenizer(query, return_tensors="pt").input_ids
        outputs = t5_model.generate(input_ids, max_new_tokens=1000)
        answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        answer = "ERROR: Model not supported yet."
    return visualized_img, chart_table, answer


title = "ChartLLM Demo v1.0"
description = "This is the **description** of ChartEQA Demo v1.0"
example_1 = ["default.png", "Example Question"]
example_2 = ["default.png", "Example Question 2"]

demo = gr.Interface(
    fn=ChartQA,
    inputs=[
        # gr.Image(type="pil"),
        gr.Image(type="pil", label = "Chart", info = "Please upload your chart image here."),
        gr.Textbox(label = "Query", info = "Please input your query here."),
        gr.Radio(["T5"], label = "Model", info = "Please choose the model you want to use.")
    ],
    outputs=[
        gr.Image(label="Detected Key Points (Blue) and Center Points (Red)", show_label=True),
        gr.Textbox(label="Converted Table", show_label=True),
        gr.Textbox(label="Answer", show_label=True).style(show_copy_button=True),
    ],
 #   examples=[example_1, example_2],
    cache_examples=True,
    allow_flagging="never",
    title=title,
    #description=description,
    theme=gr.themes.Monochrome()
)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    demo.launch(share=True)
