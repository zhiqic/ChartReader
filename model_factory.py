import torch
import importlib
import torch.nn as nn
from config import system_configs

torch.manual_seed(317)

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        module_file = "models.{}".format(system_configs.snapshot_name)
        # 使用Python的importlib库从给定的模块文件路径动态导入一个模块。
        # 动态导入允许在运行时决定导入哪个模块，而不是在编码时硬编码。这增加了代码的灵活性，允许更容易地插入插件、更改配置或根据需要加载不同的实现。
        nnet_module = importlib.import_module(module_file)
        print("Import complete")
        self.model   = nn.DataParallel(nnet_module.Model())
        print("Initiating losses")
        self.loss_function  = nnet_module.loss

        total_params = 0
        for params in self.model.parameters():
            num_params = 1
            for x in params.size():
                num_params *= x
            total_params += num_params
        print(f"Total parameters: {total_params}")

        if system_configs.opt_algo == "adam":
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters())
            )
        elif system_configs.opt_algo == "sgd":
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=system_configs.learning_rate, 
                momentum=0.9, weight_decay=0.0001
            )
        else:
            raise ValueError("Invalid optimizer specified")

    def forward(self, xs, ys, **kwargs):
        preds = self.model(*xs, **kwargs)
        loss = self.loss_function(preds, ys, **kwargs)
        return loss

    def train_step(self, xs, ys):
        self.train()
        self.optimizer.zero_grad()
        loss = self(xs, ys)
        loss = loss.mean()
        return loss

    def validate_step(self, xs, ys):
        self.eval()
        with torch.no_grad():
            loss = self(xs, ys)
            return loss.mean()

    def test(self, xs):
        self.eval()
        with torch.no_grad():
            return self.model(*xs)
    # 学习率是一个正数，用于控制模型参数在训练过程中的更新幅度。在梯度下降优化算法中，学习率与梯度的乘积确定了每次迭代中参数的更新量。
    # 较高的学习率会导致参数更新得更快，可能使训练速度加快，但也可能造成震荡和不稳定。较低的学习率会使参数更新得更慢，可能使训练更稳定，但可能会陷入局部最小值或导致训练速度变慢。
    def set_lr(self, lr):
        print(f"Setting learning rate to: {lr}")
        # 遍历 self.optimizer 中的所有参数组。在 PyTorch 中，优化器可能会有多个参数组，每个参数组可以有不同的学习率和其他超参数。
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def load_pretrained_model(self, pretrained_model):
        print("Loading from {}".format(pretrained_model))
        self.model.load_state_dict(torch.load(pretrained_model), strict=False)

    def load_model(self, iteration):
        cache_file = system_configs.snapshot_file.format(iteration)
        print(f"Loading model from {cache_file}")
        self.model.load_state_dict(torch.load(cache_file))

    def save_model(self, iteration):
        cache_file = system_configs.snapshot_file.format(iteration)
        print(f"Saving model to {cache_file}")
        torch.save(self.model.state_dict(), cache_file)