import os
import json
import torch
import torch.backends.cudnn
import argparse
import importlib
import traceback
import re
from tqdm import tqdm
from config import system_configs
from nnet.py_factory import NetworkFactory
from db.datasets import datasets
import time
torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True
import wandb
import logging
import ptvsd
#ptvsd.enable_attach(address=('0.0.0.0', 5678))
#ptvsd.wait_for_attach()

def train(training_db, validation_db, start_iter=0):
    learning_rate    = system_configs.learning_rate
    max_iter    = system_configs.max_iter
    pretrained_model = system_configs.pretrain
    snapshot         = system_configs.snapshot
    val_iter         = system_configs.val_iter
    decay_rate       = system_configs.decay_rate
    stepsize         = system_configs.stepsize
    val_ind = 0
    print("Initializing model...")
    nnet = NetworkFactory(training_db)
    wandb.watch(nnet.model, log_freq=100)

    if pretrained_model is not None:
        if not os.path.exists(pretrained_model):
            raise ValueError("The requested pretrained model does not exist.")
        print("Loading pretrained model...")

        nnet.load_pretrained_params(pretrained_model)
    print("Loading data sampling function...")
    data_file = "sampling_functions.{}".format(training_db.data)
    sample_data = importlib.import_module(data_file).sample_data
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
                nnet.load_params(start_iter)
            else:
                start_iter = 0
        nnet.set_lr(learning_rate)
        print(f"Starting training from iter {start_iter + 1}, LR: {learning_rate}...")
    else:
        nnet.set_lr(learning_rate)
    print("Training initialized...")
    nnet.cuda()
    nnet.train_mode()
    total_training_loss = []
    ind = 0
    error_count = 0

    for iteration in tqdm(range(start_iter + 1, max_iter + 1)):
        try:
            training, ind = sample_data(training_db, ind)
            #print('Data sampling OK')
            training_loss = nnet.train(**training)
            total_training_loss.append(training_loss.item())
        except:
            print('Data extraction error occurred.')
            traceback.print_exc()
            error_count += 1
            if error_count > 10:
                logging.error('Too many extraction errors. Terminating...')
                time.sleep(1)
                break
            continue
        # 使用了 Python 的参数解包（argument unpacking） 语法（**training），这意味着 training 是一个字典，其键值对会被解包并传递给 train 方法作为命名参数。
        # 假设我们有一个函数和一个字典：
        # def my_function(a, b, c):
            # return a + b + c

        # my_dict = {'a': 1, 'b': 2, 'c': 3}
        # 使用 ** 运算符，我们可以这样调用函数：
        # result = my_function(**my_dict)  # 等同于 my_function(a=1, b=2, c=3)
        training_loss = nnet.train(**training)
        total_training_loss.append(training_loss.item())

        if iteration % 500 == 0:
            avg_training_loss = sum(total_training_loss) / len(total_training_loss)
            print(f"Training loss at iter {iteration}: {avg_training_loss}")
            wandb.log({"train_loss":training_loss.item()})
            total_training_loss = []

        if val_iter and validation_db.db_inds.size and iteration % val_iter == 0:
            nnet.eval_mode()
            validation, val_ind = sample_data(validation_db, val_ind)
            validation_loss = nnet.validate(**validation)
            wandb.log({"val_loss":validation_loss.item()})
            print(f"Validation loss at iter {iteration}: {validation_loss.item()}")
            nnet.train_mode()

        if iteration % snapshot == 0:
            nnet.save_params(iteration)

        if iteration % stepsize == 0:
            learning_rate /= decay_rate
            nnet.set_lr(learning_rate)

def parse_args():
    parser = argparse.ArgumentParser(description="Train the model with given configs.")

    parser.add_argument("--cfg_file",
                        dest="cfg_file",
                        help="Specify the configuration file to be used for training.",
                        default="KPGrouping",
                        type=str)

    parser.add_argument("--iter",
                        dest="start_iter",
                        help="Specify the iter to start training from. Default is 0.",
                        default=0,
                        type=int)

    parser.add_argument("--pretrain_model",
                        dest="pretrain_model",
                        help="Specify the pre-trained model file to use. Default is 'KPDetection.pkl'.",
                        default="KPDetection.pkl",
                        type=str)

    parser.add_argument('--cache_path',
                        dest="cache_path",
                        help="Specify the cache path.",
                        type=str)

    parser.add_argument("--data_dir",
                        dest="data_dir",
                        help="Specify the directory where the data is located. Default is './data'.",
                        default="./data",
                        type=str)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    #logging.basicConfig(
        #format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        #datefmt="%m/%d/%Y %H:%M:%S",
        #level=
    #)
    args = parse_args()
    wandb.init(
        project = "ChartLLM-Extraction-Part",
        name = "chartllm-extraction-1",
        group = "chartllm-extraction",
        notes = "Test KP Detection",
        tags = ["ChartLLM", "KP Detection"],
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
        configs["system"]["pretrain"] = os.path.join(os.path.join(args.cache_path, 'nnet/KPDetection'), args.pretrain_model)
    system_configs.update_config(configs["system"])

    train_split = system_configs.train_split
    val_split   = system_configs.val_split

    print("Loading all datasets...")
    dataset = system_configs.dataset
    config_db = configs["db"]
    #print(f"configs db is {config_db}")
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