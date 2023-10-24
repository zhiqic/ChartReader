import gradio as gr
import logging
import os
from transformers import(
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration
)

t5_tokenizer = AutoTokenizer.from_pretrained("../models/t5/output/checkpoint-8000")
t5_model = T5ForConditionalGeneration.from_pretrained("../models/t5/output/checkpoint-8000")

def Ocr(chart_img):
    return None

def ChartQA(chart_img, question, model):
    chart_table = Ocr(chart_img)
    if(model == "T5"):
        question = "Question: " + question
        chart_info = "Table: " + chart_table
        query = question + chart_info
        print(query)
        input_ids = t5_tokenizer(query, return_tensors="pt").input_ids
        outputs = t5_model.generate(input_ids, max_new_tokens=1000)
        return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        return "ERROR: Model not supported yet."
    return chart_img, converted_table, answer


title = "LLMChartEnhancer Demo v1.0"
description = "This is the **description** of ChartEQA Demo v1.0"
example_1 = ["default.png", "Example Question"]
example_2 = ["default.png", "Example Question 2"]

demo = gr.Interface(
    fn=ChartQA,
    inputs=[
        # gr.Image(type="pil"),
        gr.Image(label = "Chart", info = "Please upload your chart image here."),
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
