import logging
import os
import json
import torch
import argparse
import matplotlib
matplotlib.use("Agg")
import cv2
from tqdm import tqdm
from config import system_configs
from nnet.py_factory import NetworkFactory
from db.datasets import datasets
import importlib
import json
torch.backends.cudnn.benchmark = False

def load_net(test_iter, config_name, data_dir, cache_dir, cuda_id=0):
    print(f"Loading {config_name} model")
    config_file = os.path.join(system_configs.config_dir, config_name + ".json")
    with open(config_file, "r") as f:
        configs = json.load(f)
    print(f"Configuration file loading complete")
    configs["system"]["snapshot_name"] = config_name
    configs["system"]["data_dir"] = data_dir
    configs["system"]["cache_dir"] = cache_dir
    configs["system"]["dataset"] = "Chart"
    system_configs.update_config(configs["system"])

    train_split = system_configs.train_split
    val_split = system_configs.val_split
    test_split = system_configs.test_split
    print("Config loading complete")
    split = {
        "training": train_split,
        "validation": val_split,
        "testing": test_split
    }["validation"]

    test_iter = system_configs.max_iter if test_iter is None else test_iter
    print(f"Loading parameters at iteration: {test_iter}")
    dataset = system_configs.dataset
    db = datasets[dataset](configs["db"], split)
    print("Building neural network...")
    nnet = NetworkFactory(db)
    print("Loading parameters...")
    nnet.load_params(test_iter)
    if torch.cuda.is_available():
        print("GPU available. Switching to GPU.")
        nnet.cuda(cuda_id)
    nnet.eval_mode()
    return db, nnet

# model_type: 模型的类型，如 'KPDetection' 或 'KPGrouping'。
# id_cuda: 指定是否使用 CUDA（通常用于 GPU 计算）。
# data_dir: 数据目录的路径。
# cache_dir: 缓存目录的路径。
# iteration: 迭代次数，可能用于加载模型的特定版本。
def pre_load_nets(model_type, id_cuda, data_dir, cache_dir, iteration):
    # 初始化一个空的字典 methods，用于存储预加载的模型和测试方法。
    methods = {}
    print(f"Preloading {model_type} model")
    db, nnet = load_net(iteration, model_type, data_dir, cache_dir, id_cuda)
    # 构建一个导入路径，该路径指向包含测试方法的模块。
    path = f"testfile.test_{model_type}"
    # 使用 importlib.import_module() 来动态导入该模块，并从中获取 testing 方法。
    testing = importlib.import_module(path).testing
    methods[model_type] = [db, nnet, testing]
    return methods

# 用于从给定的关键点和中心点中识别和组织图像中的组
# keys: 关键点列表。
# cens: 中心点列表。
# group_scores: 组分数矩阵。
def get_groups(keys, cens, group_scores):
    # 设置阈值，用于过滤关键点和中心点。
    thres = 0.4
    #  通过阈值过滤关键点和中心点。
    # keys[1]: 从 keys 列表中获取索引为1的元素。假设 keys 是一个包含子列表的列表，keys[1] 就是其中的第二个子列表。
    # p[0] > thres: 这是过滤条件，其中 p 是 keys[1] 中的一个元素（例如一个坐标点或分数），p[0] 是该元素的第一个值。只有当此值大于预定义阈值 thres 时，该元素才会被包括在新列表中。
    keys_trim = [p for p in keys[1] if p[0] > thres]
    cens_trim = [p for p in cens[1] if p[0] > thres]
    groups = []
    group_thres = 0.5
    # 初始化组列表和组阈值。
    if img_type == "Line":
        if len(cens_trim) == 0 or len(keys_trim) < 2: return []
        group_scores = group_scores[:len(cens_trim), len(cens_trim) :len(keys_trim)+len(cens_trim)]
        # 遍历中心点，并根据分数将关键点组织成组。
        for i in range(len(cens_trim)):
            group = []
            vals = []
            cen = cens_trim[i]
            group += [cen[2],cen[3]]
            for j in range(len(keys_trim)):
                val = group_scores[i][j].item()
                if val > 0.5:
                    key = keys_trim[j]
                    group += [key[2],key[3]]
                    vals.append(val)
            if len(vals) == 0: continue
            group.append(sum(vals)/len(vals))
            groups.append(group)
        return groups
    
    if img_type == "Bar":
        # 如果 cens_trim 为空或 keys_trim 长度小于2，则返回空列表。这可能是为了确保有足够的数据来继续处理。
        if len(cens_trim) == 0 or len(keys_trim) < 2: return []
        # 截取 group_scores 矩阵的一部分，可能与集中度和关键点有关。
        # 行索引：[:len(cens_trim)] - 这部分选择了矩阵的前 len(cens_trim) 行，其中 cens_trim 可能表示集中点或中心点的一个子集。
        # 列索引：[len(cens_trim) : len(keys_trim) + len(cens_trim)] - 这部分选择了从 len(cens_trim) 到 len(keys_trim) + len(cens_trim) 的列，其中 keys_trim 可能表示关键点的一个子集。
        group_scores = group_scores[:len(cens_trim), len(cens_trim) :len(keys_trim)+len(cens_trim)]
        # 使用 PyTorch 的 topk 函数从 group_scores 中选择前2个最大值，并获取它们的值和索引。
        vals, inds = torch.topk(group_scores, 2)
    elif img_type == "Pie":
        if len(cens_trim) == 0 or len(keys_trim) < 3: return []
        group_scores = group_scores[:len(cens_trim), len(cens_trim) :len(keys_trim)+len(cens_trim)]
        vals, inds = torch.topk(group_scores, 3)
        group_thres = 0.1
        
    for i in range(len(cens_trim)):
        # 如果当前值大于组阈值的数量等于 vals 的第二维大小
        if (vals[i] > group_thres).sum().item() == vals.size(1):
            group = []
            cen = cens_trim[i]
            group += [cen[2],cen[3]]
            for ind in inds[i]:
                key = keys_trim[ind]
                group += [key[2],key[3]]
            group.append(vals[i].mean().item())
            groups.append(group)

    return groups

    
def test(image_path, model_type):
    image = cv2.imread(image_path)
    # 使用 PyTorch 的 torch.no_grad() 上下文管理器来禁用梯度计算，以提高推理速度并减少内存使用。
    with torch.no_grad():
        # 使用预加载的 'KPDetection' 方法（存储在 methods 字典中）对图像进行关键点检测。methods['KPDetection'][2] 是测试函数，methods['KPDetection'][0] 是数据库对象，methods['KPDetection'][1] 是神经网络对象。
        results = methods[model_type][2](image, methods[model_type][0], methods[model_type][1], debug=False)
        if model_type == 'KPDetection':
            # 从 results 中提取关键点（keys）和中心点（centers）。
            keys, centers = results[0], results[1]
            thres = 0.
            # 从 results 中提取关键点（keys）和中心点（centers）。
            keys = {k: [p for p in v.tolist() if p[0]>thres] for k,v in keys.items()} 
            centers = {k: [p for p in v.tolist() if p[0]>thres] for k,v in centers.items()}
            
            return (keys, centers)
        if model_type == 'KPGrouping':
            keys, centers, group_scores = results
            # 与 'KPDetection' 相同，但这里没有应用阈值过滤。
            keys = {k: [p for p in v.tolist()] for k,v in keys.items()} 
            centers = {k: [p for p in v.tolist()] for k,v in centers.items()}
            groups = get_groups(keys, centers, group_scores)
            
            return (keys, centers, groups)

def parse_args():
    parser = argparse.ArgumentParser(description="Test the ChartReader extraction part.")

    parser.add_argument("--img_path",
                        dest="img_path",
                        help="Path to the images to be tested. Default is 'data/chart/images/val'.",
                        default="data/chart/images/val",
                        type=str)

    parser.add_argument("--save_path",
                        dest="save_path",
                        help="Path to save the test results. Default is 'tmp/'.",
                        default="tmp/",
                        type=str)

    parser.add_argument("--model_type",
                        dest="model_type",
                        help="Specify the type of model to use for testing. Default is 'Grouping'.",
                        default="Grouping",
                        type=str)

    parser.add_argument("--iter",
                        dest="iter",
                        help="Specify the number of iterations the model was trained for. Default is '50000'.",
                        default='50000',
                        type=str)

    parser.add_argument("--data_dir",
                        dest="data_dir",
                        help="Path to the directory where the data is located. Default is 'data/chart/'.",
                        default="data/chart/",
                        type=str)

    parser.add_argument('--cache_path',
                        dest="cache_path",
                        help="Specify the cache path. Default is 'data/chart/cache/'.",
                        default="data/chart/cache/",
                        type=str)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    methods = pre_load_nets(args.model_type, 0, args.data_dir, args.cache_path, args.iter)
    save_path = os.path.join(args.save_path, args.model_type + args.iter + '.json')
    # 初始化一个空字典，用于存储图片和它们的预测结果
    rs_dict = {}
    # 列出指定文件夹（args.img_path）内的所有图片
    images = os.listdir(args.img_path)
    print(f"Predicting with {args.model_type} net")
    for img in tqdm(images):
        path = os.path.join(args.img_path, img)
        data = test(path, args.model_type)
        rs_dict[img] = data
    with open(save_path, "w") as f:
        json.dump(rs_dict, f)