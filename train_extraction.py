import os
import json
import torch
import torch.backends.cudnn
import argparse
from sampling_function import sample_data
import traceback
import re
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from socket import error as SocketError
import errno
from tqdm import tqdm
from config import system_configs
from model_factory import Network
from db.datasets import datasets
import time
from torch.multiprocessing import Process, Queue
torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True
import wandb

#import ptvsd
#ptvsd.enable_attach(address=('0.0.0.0', 5678))
#ptvsd.wait_for_attach()

# db: 数据库或数据集对象，用于从中采样数据。
# queue: 用于存放预取数据的队列。
# sample_data: 一个函数，用于从数据库或数据集中采样数据。应该接受数据库、当前索引和数据增强选项作为输入，并返回采样的数据和下一个索引。
# data_aug: 可能的数据增强选项或参数。
def prefetch_data(db, queue, sample_data, data_aug):
    ind = 0
    print("Starting data prefetching process...")
    # 设置随机种子，使每个进程的随机数生成器独立。使用进程ID作为种子。
    np.random.seed(os.getpid())
    while True:
        try:
            # 调用 sample_data 函数，从数据库或数据集中采样数据，并获取下一个索引。
            data, ind = sample_data(db, ind, data_aug=data_aug)
            queue.put(data)
        except Exception as e:
            print(f'An error occurred during data prefetching: {e}')
            traceback.print_exc()

# 用于在数据加载过程中将张量固定（pin）到内存中。在使用GPU训练时，固定内存可以加速数据从CPU到GPU的传输。
# data_queue: 包含待处理数据的队列。
# pinned_data_queue: 用于存放固定内存后的数据的队列。
# sema: 信号量，用于同步或控制线程。            
def pin_memory(data_queue, pinned_data_queue, sema):
    while True:
        try:
            data = data_queue.get()
            # 对字典中的 "xs" 键对应的每个张量使用 pin_memory() 方法，将其固定到内存中。
            data["xs"] = [x.pin_memory() for x in data["xs"]]
            data["ys"] = [y.pin_memory() for y in data["ys"]]
            pinned_data_queue.put(data)
            # 尝试获取信号量（非阻塞方式）。如果成功获取，则退出循环并返回。这可以用于控制何时停止线程。
            if sema.acquire(blocking=False):
                return
        # 捕获套接字错误。
        except SocketError as e:
            # 如果错误不是连接重置错误，则重新引发异常。
            if e.errno != errno.ECONNRESET:
                raise
            pass

# dbs: 数据库或数据集的列表，每个数据库/数据集用于一个单独的进程。
# queue: 用于存放预取数据的队列。
# fn: 一个函数，用于从数据库或数据集中采样数据。
# data_aug: 可能的数据增强选项或参数。
def init_parallel_jobs(dbs, queue, fn, data_aug):
    # 对于 dbs 列表中的每个数据库/数据集，创建一个新的进程对象。目标函数是 prefetch_data，并传递相应的参数。
    tasks = [Process(target=prefetch_data, args=(db, queue, fn, data_aug)) for db in dbs]
    for task in tasks:
        # 将进程设置为守护进程。守护进程是在后台运行的进程，当主程序结束时，它们也会被终止。
        task.daemon = True
        task.start()
    return tasks

def train(training_db, validation_db, start_iter=0):
    learning_rate    = system_configs.learning_rate
    max_iter    = system_configs.max_iter
    pretrained_model = system_configs.pretrain
    val_iter         = system_configs.val_iter
    decay_rate       = system_configs.decay_rate
    stepsize         = system_configs.stepsize
    val_ind = 0
    print("Initializing model...")
    nnet = Network()
    #wandb.watch(nnet.model, log_freq=100)
    if pretrained_model is not None:
        print(pretrained_model)
        if not os.path.exists(pretrained_model):
            raise ValueError("The requested pretrained model does not exist.")
        print("Loading pretrained model...")

        nnet.load_pretrained_model(pretrained_model)
    print("Loading data sampling function...")
    if start_iter:
        if start_iter == -1:
            print("Training from latest iter...")
            save_list = os.listdir(system_configs.snapshot_dir)
            save_list = [f for f in save_list if f.endswith('.pkl')]
            save_list.sort(reverse=True, key = lambda x: int(x.split('_')[1][:-4]))
            if len(save_list) > 0:
                target_save = save_list[0]
                start_iter = int(re.findall(r'\d+', target_save)[0])
                learning_rate /= (decay_rate ** (start_iter // stepsize))
                nnet.load_model(start_iter)
            else:
                start_iter = 0
        nnet.set_lr(learning_rate)
        print(f"Starting training from iter {start_iter + 1}, LR: {learning_rate}...")
    else:
        nnet.set_lr(learning_rate)
    print("Training initialized...")
    total_training_loss = []
    ind = 0
    error_count = 0
    scaler = GradScaler()
    optimizer = nnet.optimizer  # 确保你的模型有一个优化器属性
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nnet.to(device)
    best_val_loss = float('inf')
    for iteration in tqdm(range(start_iter + 1, max_iter + 1)):
        try:
            training, ind = sample_data(training_db, ind)
            training_data = []
            for d in training.values():
                if isinstance(d, torch.Tensor):
                    training_data.append(d.to(device))
                elif isinstance(d, list):
                    training_data.append([item.to(device) if isinstance(item, torch.Tensor) else item for item in d])
                else:
                    training_data.append(d)

            optimizer.zero_grad()

            # 使用 autocast
            with autocast():
                training_loss = nnet.train_step(*training_data)
        
            # 缩放梯度
            scaler.scale(training_loss).backward()
        
            # 调用 scaler.step() 来更新权重
            scaler.step(optimizer)
        
            # 更新缩放器
            scaler.update()

            total_training_loss.append(training_loss.item())
        except:
            print('Data extraction error occurred.')
            traceback.print_exc()
            error_count += 1
            if error_count > 10:
                print('Too many extraction errors. Terminating...')
                time.sleep(1)
                break
            continue

        if iteration % 500 == 0:
            avg_training_loss = sum(total_training_loss) / len(total_training_loss)
            print(f"Training loss at iter {iteration}: {avg_training_loss}")
            wandb.log({"train_loss":training_loss.item()})
            total_training_loss = []

        if val_iter and validation_db.db_inds.size and iteration % val_iter == 0:
            validation, val_ind = sample_data(validation_db, val_ind)
            validation_data = []
            for d in validation.values():
                if isinstance(d, torch.Tensor):
                    validation_data.append(d.to(device))
                elif isinstance(d, list):
                    validation_data.append([item.to(device) if isinstance(item, torch.Tensor) else item for item in d])
                else:
                    validation_data.append(d)
            validation_loss = nnet.validate_step(*validation_data)
            wandb.log({"val_loss":validation_loss.item()})
            print(f"Validation loss at iter {iteration}: {validation_loss.item()}")
            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                print(f"New best validation loss: {best_val_loss.item()}. Saving model...")
                nnet.save_model("best")

        if iteration % stepsize == 0:
            learning_rate /= decay_rate
            nnet.set_lr(learning_rate)

def parse_args():
    parser = argparse.ArgumentParser(description="Train the model with the given configs.")

    parser.add_argument("--cfg_file",
                        dest="cfg_file",
                        help="Name of the configuration file to be used for training.",
                        default="KPDetection",
                        type=str)

    parser.add_argument("--start_iter",
                        dest="start_iter",
                        help="Specify the iter to start training from. Default is 0.",
                        default=0,
                        type=int)

    parser.add_argument("--pretrained_model",
                        dest="pretrained_model",
                        help="Name of the pre-trained model file. Default is 'KPDetection.pkl'.",
                        default="KPDetection.pkl",
                        type=str)

    parser.add_argument("--threads",
                        dest="threads",
                        help="Number of threads to use for data loading. Default is 1.",
                        default=1,
                        type=int)

    parser.add_argument("--cache_path",
                        dest="cache_path",
                        help="Path to cache preprocessed data for faster loading and to save trained models.",
                        default="./data/cache/",
                        type=str)

    parser.add_argument("--data_dir",
                        dest="data_dir",
                        help="Directory containing the dataset for training. Default is './data'.",
                        default="./data",
                        type=str)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    wandb.init(
        project = "ChartLLM-Extraction",
        name = "bar only",
        group = "grouping",
        notes = "Test KP Grouping with Only Bars-Mixed Precision-No Crop or Bump",
        tags = ["ChartLLM", "KP Grouping"],
        config = args
    )
    print(f"Training args: {args}")
    cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
    with open(cfg_file, "r") as f:
        configs = json.load(f)
    configs["system"]["data_dir"] = args.data_dir
    configs["system"]["cache_dir"] = args.cache_path
    configs["system"]["dataset"] = "Chart"
    file_list_data = os.listdir(args.data_dir)
    # print(file_list_data)
    configs["system"]["snapshot_name"] = args.cfg_file
    if args.cfg_file == "KPGrouping":
        if(args.start_iter == 0):
            configs["system"]["pretrain"] = os.path.join(os.path.join(args.cache_path, 'nnet/KPDetection'), args.pretrained_model)
        else:
            configs["system"]["pretrain"] = os.path.join(os.path.join(args.cache_path, 'nnet/KPGrouping'), args.pretrained_model)
    else:
        if(args.start_iter != 0):
            configs["system"]["pretrain"] = os.path.join(os.path.join(args.cache_path, 'nnet/KPDetection'), args.pretrained_model)
    system_configs.update_config(configs["system"])

    train_split = system_configs.train_split
    val_split   = system_configs.val_split

    print("Loading all datasets...")
    dataset = system_configs.dataset
    threads = args.threads
    print(f"Using {threads} threads.")
    training_db  = datasets[dataset](configs["db"], train_split) 
    validation_db = datasets[dataset](configs["db"], val_split)

    print("Current system configuration:")
    print(system_configs.full)

    print("Current database configuration:")
    print(training_db.configs)

    print(f"Number of indices in training database: {len(training_db.db_inds)}")
    #print(training_db.db_inds)
    #for i in training_db.db_inds:
        #print(training_db.image_ids(i))
    print(f"Number of indices in validation database: {len(validation_db.db_inds)}")
    train(training_db, validation_db, args.start_iter)
