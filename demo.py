#!/root/miniconda3/envs/ChartLLM/bin/python
import gradio as gr
import logging
import os
from transformers import(
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration
)
import subprocess
import json
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
#t5_tokenizer = AutoTokenizer.from_pretrained("../models/t5/output/checkpoint-8000")
#t5_model = T5ForConditionalGeneration.from_pretrained("../models/t5/output/checkpoint-8000")

def Ocr(chart_img):
    return None

from PIL import Image, ImageDraw

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
    thres = 0.1
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
    #chart_table = Ocr(chart_img)
    #if(model == "T5"):
        #question = "Question: " + question
        #chart_info = "Table: " + chart_table
        #query = question + chart_info
        #print(query)
        #input_ids = t5_tokenizer(query, return_tensors="pt").input_ids
        #outputs = t5_model.generate(input_ids, max_new_tokens=1000)
        #return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    #else:
        #return "ERROR: Model not supported yet."
    return visualized_img, chart_type, "Test"


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
