# ChartLLM

ChartLLM is a sophisticated implementation for the interpretation and understanding of various types of data charts.

## System Requirements

- GPU-enabled machine
- CUDA toolkit version = 10.0
- Python 3.11
- GCC 4.9.2 or above
  
## Installation

### Prerequisites

Ensure [Anaconda](https://anaconda.org) is installed on your system. Use the provided package list to create an Anaconda environment:

```shell
conda env create -f requirements.yaml
conda activate ChartLLM
```

### Compiling NMS

Compile the NMS code required for the chart data extraction part, which are originally from [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/cpu_nms.pyx) and [Soft-NMS](https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/cpu_nms.pyx).

```shell
cd external
make
```

### Installing MS COCO APIs

The Microsoft COCO APIs are also required for the functioning of the chart data extraction part.

```shell
mkdir data
cd data
git clone https://github.com/cocodataset/cocoapi coco
cd coco/PythonAPI
make
```

## Data Preparation

Download the modified EC400K dataset from this link 

The annotation contains three parts. 

The first part `images` containes the chart image information, which has 4 labels for each image: `file_name`, `width`, `height`, and `id`. 

The second part `annotations` contains the chart components annotation information, which has 5 labels for each annotation: `image_id`, `category_id`,   `bbox`, `area`, `id`.

- `image_id`: the `id` of chart image which the annotation belongs to
- `category_id`: the type of the component, where 1 denotes bars in bar charts, 2 denotes lines in line charts, 3 denotes pies in pie charts, 4 denotes the legends, 5 denotes the title of the values axes, 6 denotes the title of the entire chart, 7 denotes the title of the category axes.
- `bbox`: the points showing the bounding box of the component. For lines in line charts, they are the data points for a line (`[d_1_x, d_1_y, …., d_n_x, d_n_y]`). For pies in pie charts, they are the three critical points for a sector of the pie `[center_x, center_y, edge_1_x, edge_1_y, edge_2_x, edge_2_y]`. For bars in bar charts, and other types of components, they are the x-coordinate of the top-left corner of the box, the y-coordinate of the top-left corner of the box, the width of the box (horizontal dimension), and the height of the box (vertical dimension).
- `area`: the area of the chart component.
- `id`: the unique identifier of each annotation.

The third part `categories` provide a further reference of the `categories` in the `annotations` part with 3 labels for each component category: `supercategory`, `id`, and `name`. In which categories 1 to 3 belong to supercategory "MainComponent" and other categories belong to supercategory "OtherComponents".

## Training 

### Chart Data Extraction Part

The configuration files `KPDetection` for keypoint detection and `KPGrouping` for keypoint detection and grouping are in JSON format and located in `config/`.

To train the chart data extraction model, use the `train_extraction.py` script. You should first train the KP Detection model, for example:

```shell
python train_extraction.py \
    --cfg_file KPDetection \
    --data_dir "data/extraction_data/" \
    --cache_path "data/cache/" \
    --pretrain_model "KPDetection_20000.pkl" \
    --iter 20000
```

Then you can use the pretrained KP Detection model to train the KP grouping model, for example:

```shell
python train_extraction.py \
    --cfg_file KPGrouping \
    --data_dir "data/extraction_data/" \
    --pretrain_model "KPGrouping_20000.pkl" \
    --cache_path "data/cache/" \
    --iter 20000
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
        --train_file="./data/qa_data/train.csv" \
        --validation_file="./data/qa_data/val.csv" \
        --test_file="./data/qa_data/test.csv" \
        --text_column=Input \
        --summary_column=Output \
        --source_prefix="" \
        --output_dir="./data/cache/t5_output" \
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

### Chart Data Extraction Part

To test batches of data directly, use the `val.py` script:

```shell
python val_extraction.py \
    --img_path <path_to_test_images> \
    --save_path <path_to_save_test_results> \
    --model_type <type_of_model_to_be_tested(KPDetection/KPGrouping)> \
    --cache_path <path_to_model_to_be_tested> \
    --data_dir <path_to_data> \
    --iter <model_iteration>
```

e.g.

```shell
python val_extraction.py \
    --img_path "data/extraction_data/images/val" \
    --save_path evaluation \
    --model_type KPGrouping \
    --cache_path "data/cache/" \
    --data_dir "data/extraction_data" \
    --iter 15000
```

## Todo

End to end training