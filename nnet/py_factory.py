import torch
import importlib
import torch.nn as nn
import logging
from config import system_configs

torch.manual_seed(317)

class Network(nn.Module):
    def __init__(self, model, loss):
        super().__init__()

        self.model = model
        self.loss  = loss

    def forward(self, xs, ys, **kwargs):
        preds = self.model(*xs, **kwargs)
        loss  = self.loss(preds, ys, **kwargs)
        return loss

# for model backward compatibility
# previously model was wrapped by DataParallel module
class DummyModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model

    def forward(self, *xs, **kwargs):
        return self.module(*xs, **kwargs)

class NetworkFactory():
    def __init__(self, db):
        super().__init__()

        module_file = "models.{}".format(system_configs.snapshot_name)
        # 使用Python的importlib库从给定的模块文件路径动态导入一个模块。
        # 动态导入允许在运行时决定导入哪个模块，而不是在编码时硬编码。这增加了代码的灵活性，允许更容易地插入插件、更改配置或根据需要加载不同的实现。
        nnet_module = importlib.import_module(module_file)
        print("Import Complete")
        self.model   = DummyModule(nnet_module.Model(db))
        print("Initiating Losses")
        self.loss    = nnet_module.loss
        self.network = Network(self.model, self.loss)

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

    # 将模型和网络移动到指定的 CUDA 设备（通常是 GPU）上
    def cuda(self, cuda_id=0):
        # 调用模型的 cuda 方法，并传入 cuda_id 作为参数。这会将模型的所有参数和缓冲区移动到指定的 CUDA 设备上。
        self.model.cuda(cuda_id)
        # 将另一个名为 self.network 的网络结构移动到指定的 CUDA 设备上。
        self.network.cuda(cuda_id)
        self.cuda_id = cuda_id

    def train_mode(self):
        self.network.train()

    def eval_mode(self):
        self.network.eval()

    def train(self, xs, ys, **kwargs):
        #print("Train function being called")
        xs = [x.cuda(non_blocking=True, device=self.cuda_id) for x in xs]
        ys = [y.cuda(non_blocking=True, device=self.cuda_id) for y in ys]
        # for i, tensor in enumerate(xs):
            # print(f"The shape of tensor at xs index {i} is {tensor.shape}")
        # for i, tensor in enumerate(ys):
            # print(f"The shape of tensor at ys index {i} is {tensor.shape}")
        # print(f"xs: {xs}")
        # print(f"ys: {ys}")
        self.optimizer.zero_grad()
        loss = self.network(xs, ys)
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()
        return loss

    def validate(self, xs, ys, **kwargs):
        with torch.no_grad():
            if torch.cuda.is_available():
                xs = [x.cuda(non_blocking=True, device=self.cuda_id) for x in xs]
                ys = [y.cuda(non_blocking=True, device=self.cuda_id) for y in ys]

            loss = self.network(xs, ys)
            loss = loss.mean()
            return loss

    def test(self, xs, **kwargs):
        with torch.no_grad():
            if torch.cuda.is_available():
                # non_blocking=True: 这是一个优化，允许其他操作和数据迁移同时进行。
                # for x in xs: 这个循环遍历所有输入数据（通常是一个批次的数据），并将它们移动到 GPU。
                xs = [x.cuda(non_blocking=True, device=self.cuda_id) for x in xs]
            # 调用模型的前向传播方法，并传入输入数据和任何其他关键字参数。结果（模型的输出）将被返回。
            return self.model(*xs, **kwargs)
        
    # 学习率是一个正数，用于控制模型参数在训练过程中的更新幅度。在梯度下降优化算法中，学习率与梯度的乘积确定了每次迭代中参数的更新量。
    # 较高的学习率会导致参数更新得更快，可能使训练速度加快，但也可能造成震荡和不稳定。较低的学习率会使参数更新得更慢，可能使训练更稳定，但可能会陷入局部最小值或导致训练速度变慢。
    def set_lr(self, lr):
        print(f"Setting learning rate to: {lr}")
        # 遍历 self.optimizer 中的所有参数组。在 PyTorch 中，优化器可能会有多个参数组，每个参数组可以有不同的学习率和其他超参数。
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def load_pretrained_params(self, pretrained_model):
        print("Loading from {}".format(pretrained_model))
        with open(pretrained_model, "rb") as f:
            params = torch.load(f)
            self.model.load_state_dict(params, strict=False)

    def load_params(self, iteration):
        cache_file = system_configs.snapshot_file.format(iteration)
        print(f"Loading model from {cache_file}")
        with open(cache_file, "rb") as f:
            if torch.cuda.is_available():
                params = torch.load(f)
            else:
                params = torch.load(f, map_location='cpu')
            self.model.load_state_dict(params)

    def save_params(self, iteration):
        cache_file = system_configs.snapshot_file.format(iteration)
        print(f"Saving model to {cache_file}")
        with open(cache_file, "wb") as f:
            params = self.model.state_dict()
            torch.save(params, f)