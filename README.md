# ChartReader: A Unified Framework for Chart Derendering and Comprehension without Heuristic Rules

## Highlights

- 🏅 Easily handle unseen chart types by simply adding more training data 
- 🎊 Transformer architecture automatically infers rules from center/key points
- 🏵️ Unified framework for all chart and table understanding tasks

Our solution first uses a specialized detection module built on Multi-Scale [Hourglass Networks](https://arxiv.org/abs/1603.06937) to locate and segment chart components like axes, legends, and plot areas in a unified manner without hardcoded assumptions. We then employ a structured Transformer encoder to capture spatial, semantic, and stylistic relationships between the detected components. This allows grouping relevant elements into a structured tabular intermediate representation of the chart layout and content. Finally, we fine-tune the state-of-the-art [T5](https://arxiv.org/abs/1910.10683) text-to-text transformer on this representation using special tokens to associate chart details with free-form questions across a variety of analytical tasks.

## Quickstart

### System Requirements

- GPU-enabled machine
- CUDA toolkit version = 10.0
- Python 3.11
- GCC 4.9.2 or above
  
### Installation

Ensure [Anaconda](https://anaconda.org) is installed on your system. Use the provided package list to create an Anaconda environment:

```shell
conda env create -f requirements.yaml
conda activate ChartLLM
```

The Microsoft COCO APIs are required for the functioning of the data loading part of the chart data extraction part, given that the original EC400K dataset is in COCO format.

```shell
mkdir data
cd data
git clone https://github.com/cocodataset/cocoapi coco
cd coco/PythonAPI
make
```

### Data Preparation

Download the modified EC400K dataset from this [link](https://pan.baidu.com/s/1myO8-SAmLa5NVsHzmBSC5w?pwd=54tb)

> The annotation contains three parts. 
> The first part `images` contains the chart image information, which has 4 labels for each image: `file_name`, `width`, `height`, and `id`. 
> The second part `annotations` contains the chart components annotation information, which has 5 labels for each annotation: `image_id`, `category_id`, `bbox`, `area`, and `id`.
> - `image_id`: the `id` of chart image which the annotation belongs to
> - `category_id`: the type of the component, where 1 denotes bars in vbar_categorical charts, 2 denotes lines in line charts, 3 denotes pies in pie charts, 4 denotes the legends, 5 denotes the title of the values axes, 6 denotes the title of the entire chart, 7 denotes the title of the category axes.
> - `bbox`: the points showing the bounding box of the component. For lines in line charts, they are the data points for a line (`[d_1_x, d_1_y, …., d_n_x, d_n_y]`). For pies in pie charts, they are the three critical points for a sector of the pie `[center_x, center_y, edge_1_x, edge_1_y, edge_2_x, edge_2_y]`. For bars in vbar_categorical charts, and other types of components, they are the x-coordinate of the top-left corner of the box, the y-coordinate of the top-left corner of the box, the width of the box (horizontal dimension), and the height of the box (vertical dimension).
> - `area`: the area of the chart component.
> - `id`: the unique identifier of each annotation.
> The third part `categories` provide a further reference of the `categories` in the `annotations` part with 3 labels for each component category: `supercategory`, `id`, and `name`. In which categories 1 to 3 belong to supercategory "MainComponent" and other categories belong to supercategory "OtherComponents".

## Training 

### Chart Data Extraction Part

The configuration files `KPDetection` for keypoint detection and `KPGrouping` for keypoint detection and grouping are in JSON format and located in `config/`.

To train the chart data extraction model, use the `train_extraction.py` script. You should first train the KP Detection model, for example:

```shell
python train_extraction.py \
    --cfg_file KPDetection \
    --data_dir "/root/autodl-tmp/bar/" \
    --cache_path "/root/autodl-tmp/cache/"
```

Then you can use the pretrained KP Detection model to train the KP grouping model, for example:

```shell
python train_extraction.py \
    --cfg_file KPGrouping \
    --data_dir "/root/autodl-tmp/component_data/" \
    --pretrained_model "KPDetection_best.pkl" \
    --cache_path "/root/autodl-tmp/cache/" 
```

### Chart Question Answering Part

Execute the command below, ensuring to replace placeholder paths and adjust hyperparameters as necessary:

```shell
torchrun \
    --nproc_per_node=1 \
    run_t5.py \
        --model_name_or_path=t5-base \
        --do_train \
        --do_eval \
        --do_predict \
        --train_file="/root/autodl-tmp/qa_data/train.csv" \
        --validation_file="/root/autodl-tmp/qa_data/val.csv" \
        --test_file="/root/autodl-tmp/qa_data/test.csv" \
        --text_column=Input \
        --summary_column=Output \
        --source_prefix="" \
        --output_dir="/root/autodl-tmp/cache/t5_output" \
        --per_device_train_batch_size=8 \
        --per_device_eval_batch_size=16 \
        --predict_with_generate=True \
        --learning_rate=0.0001 \
        --num_beams=4 \
        --num_train_epochs=30 \
        --save_steps=10000 \
        --eval_steps=2000 \
        --evaluation_strategy=steps \
        --load_best_model \
        --max_source_length=1024
```

## Evaluation

To use the demo UI interface, use the `demo.py` script, ensure you have replaced all the directories in the script with correct values. 

To test the extraction of data directly, use the `val_extraction.py` script:

e.g.

```shell
python val_extraction.py \
    --save_path evaluation \
    --model_type KPGrouping \
    --cache_path "/root/autodl-tmp/cache/" \
    --data_dir "/root/autodl-tmp/component_data" \
    --trained_model_iter "best"
```

## Acknowledgments

This work received support from the Air Force Research Laboratory under agreement number FA8750-192-0200; the Defense Advanced Research Projects Agency (DARPA) grants funded through the GAILA program (award HR00111990063); and the Defense Advanced Research Projects Agency (DARPA) grants funded through the AIDA program (FA8750-18-20018).

## Citation

If you use this code, please cite the following paper:

```plaintext
@inproceedings{cheng2023chartreader,
  title={Chartreader: A unified framework for chart derendering and comprehension without heuristic rules},
  author={Cheng, Zhi-Qi and Dai, Qi and Li, Siyao and Sun, Jingdong and Mitamura, Teruko and Hauptmann, Alexander G.},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={22202--22213},
  year={2023}
}
```

## License

This project is licensed under the terms of the MIT license. It is intended for academic use only.
