import torch
import torch.nn as nn
from .utils import convolution, residual
from .utils import make_layer, make_layer_revr
from .kp_utils import _transpose_and_gather_feat, _decode_detection, _decode_group
from .kp_utils import _sigmoid, _regr_loss, _neg_loss
from .kp_utils import make_kp_layer
from .kp_utils import make_pool_layer, make_unpool_layer
from .kp_utils import make_merge_layer, make_inter_layer, make_cnv_layer

class kp_module(nn.Module):
# n: 层级数量，决定了递归构建的深度。
# dims: 一个列表，包括每个层级的维度。
# modules: 一个列表，包括每个层级的模块数量。
# layer: 用于构建残差块的层类型。
# make_xxx_layer: 一系列的函数，用于构建不同的层类型。
# 作用
# 多尺度特征提取: 通过上采样和下采样结合残差连接，kp_module能够捕捉多尺度的特征。
# 递归结构: 通过递归构建，可以灵活地创建不同深度的结构。
# 特征融合: 合并层允许来自不同路径和尺度的特征进行融合。
    def __init__(
        self, n, dims, modules, layer=residual,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, **kwargs
    ):
        super().__init__()

        self.n   = n

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

# 结构组成
# self.up1: 上采样层，用于增加特征映射的尺寸。
        self.up1  = make_up_layer(
            3, curr_dim, curr_dim, curr_mod, 
            layer=layer, **kwargs
        )  
# self.max1: 最大池化层，用于下采样。
        self.max1 = make_pool_layer()
# self.low1: 下方层，用于处理下采样后的特征。
        self.low1 = make_hg_layer(
            3, curr_dim, next_dim, curr_mod,
            layer=layer, **kwargs
        )
# self.low2: 如果层级n大于1，则递归构建另一个kp_module；否则，构建另一个下方层。
        self.low2 = kp_module(
            n - 1, dims[1:], modules[1:], layer=layer, 
            make_up_layer=make_up_layer, 
            make_low_layer=make_low_layer,
            make_hg_layer=make_hg_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_pool_layer=make_pool_layer,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer,
            **kwargs
        ) if self.n > 1 else \
        make_low_layer(
            3, next_dim, next_dim, next_mod,
            layer=layer, **kwargs
        )
# self.low3: 反向的下方层，用于处理递归结构的输出。
        self.low3 = make_hg_layer_revr(
            3, next_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )
# self.up2: 上采样层，用于恢复特征映射的尺寸。
        self.up2  = make_unpool_layer(curr_dim)
# self.merge: 合并层，用于融合不同路径的特征。
        self.merge = make_merge_layer()
    # forward 方法是在定义自定义 nn.Module 子类时需要重写（override）的方法之一。这个方法定义了如何对输入数据进行操作以得到输出。当你调用一个 nn.Module 对象时（就像函数一样），实际上是调用了它的 forward 方法。
    def forward(self, x):
        # 输入通过上采样层self.up1，将特征映射的尺寸增加。这有助于捕获更精细的空间信息。
        up1  = self.up1(x)
        # 输入通过最大池化层self.max1，将特征映射的尺寸减小。这有助于捕获更全局的特征。
        max1 = self.max1(x)
        # 池化后的特征通过下方层self.low1进行进一步转换。
        low1 = self.low1(max1)
        # 下方层的输出通过另一个下方层或递归结构self.low2。如果这是一个递归的kp_module，这个步骤将递归地应用整个流程。
        low2 = self.low2(low1)
        # 递归结构的输出通过反向下方层self.low3进行转换。
        low3 = self.low3(low2)
        # 特征通过上采样层self.up2，将尺寸恢复到与up1相同。
        up2  = self.up2(low3)
        # 上采样的输出up1和up2通过合并层self.merge进行融合，并返回最终的输出。
        return self.merge(up1, up2)

class kp_detection(nn.Module):
    def __init__(
        self, n, nstack, dims, modules, out_dim, pre=None, cnv_dim=256,
        make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
        make_regr_layer=make_kp_layer,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer,
        kp_layer=residual
    ):
        super(kp_detection, self).__init__()
        print('Keypoint detection enabled.')
        self.nstack    = nstack
        self._decode   = _decode_detection
        curr_dim = dims[0]
# 定义了一个简单的前处理（preprocessing）或初级特征提取（initial feature extraction）部分。这一部分通常用于在更深层次的特征提取和任务特定操作之前对输入图像进行初步处理。让我们逐一解析这些层的作用：
# convolution(7, 3, 128, stride=2): 这是一个卷积层，卷积核尺寸为7x7，输入通道数为3（假设是RGB图像），输出通道数为128。步长（stride）为2，这意味着这个层会降低图像的尺寸。
# residual(3, 128, 256, stride=2): 这看起来像一个残差块（Residual Block）。它接收128个通道的输入，并输出256个通道。步长为2，这意味着它也会降低特征图（feature map）的尺寸。
        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre

        self.kps  = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            ) for _ in range(nstack)
        ])
        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])

        self.key_cnvs = nn.ModuleList([
            make_cnv_layer(cnv_dim, cnv_dim) for _ in range(nstack)
        ])
        self.center_cnvs = nn.ModuleList([
            make_cnv_layer(cnv_dim, cnv_dim) for _ in range(nstack)
        ])

        # keypoint heatmaps
        self.key_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])
        self.center_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        for key_heat, center_heat in zip(self.key_heats, self.center_heats):
            # 选取 key_heat 列表中的最后一个元素（假设为一个网络层），并将该层的偏置参数全部设置为 -2.19。
            key_heat[-1].bias.data.fill_(-2.19)
            center_heat[-1].bias.data.fill_(-2.19)

        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])

        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])

        self.key_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.center_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])

        self.relu = nn.ReLU(inplace=True)

# 输入参数
# image: 输入图像数据。
# key_inds: 关键点索引。
# center_inds: 中心点索引。
    def _train(self, *xs):
        image       = xs[0]
        key_inds    = xs[1]
        center_inds = xs[2]
        #  使用预处理层（self.pre）对图像进行预处理。
        inter = self.pre(image)
        outs  = []

        layers = zip(
            self.kps, self.cnvs,
            self.key_cnvs, self.center_cnvs,
            self.key_heats, self.center_heats,
            self.key_regrs, self.center_regrs
        )
        # 遍历每一个层组合。
        for ind, layer in enumerate(layers):
            # 使用 kp_, cnv_, ... 来分解当前的层组合。
            kp_, cnv_          = layer[0:2]
            key_cnv_, center_cnv_   = layer[2:4]
            key_heat_, center_heat_ = layer[4:6]
            key_regr_, center_regr_ = layer[6:8]
            # 通过关键点层计算关键点。
            kp  = kp_(inter)
            # 对关键点应用卷积层。
            cnv = cnv_(kp)
            # 分别对关键点和中心点应用卷积。
            key_cnv = key_cnv_(cnv)
            center_cnv = center_cnv_(cnv)
            # 计算关键点和中心点的热图。
            key_heat, center_heat = key_heat_(key_cnv), center_heat_(center_cnv)
            # 计算关键点和中心点的回归。
            key_regr, center_regr = key_regr_(key_cnv), center_regr_(center_cnv)
            key_regr = _transpose_and_gather_feat(key_regr, key_inds)
            center_regr = _transpose_and_gather_feat(center_regr, center_inds)

            outs += [key_heat, center_heat, key_regr, center_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        
        return outs

    def _test(self, *xs, **kwargs):
        image = xs[0]

        inter = self.pre(image)
        outs  = []

        layers = zip(
            self.kps, self.cnvs,
            self.key_cnvs, self.center_cnvs,
            self.key_heats, self.center_heats,
            self.key_regrs, self.center_regrs
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]
            key_cnv_, center_cnv_   = layer[2:4]
            key_heat_, center_heat_ = layer[4:6]
            key_regr_, center_regr_ = layer[6:8]

            kp  = kp_(inter)
            cnv = cnv_(kp)

            if ind == self.nstack - 1:
                key_cnv = key_cnv_(cnv)
                center_cnv = center_cnv_(cnv)

                key_heat, center_heat = key_heat_(key_cnv), center_heat_(center_cnv)
                key_regr, center_regr = key_regr_(key_cnv), center_regr_(center_cnv)

                outs += [key_heat, center_heat, key_regr, center_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)

        return self._decode(*outs[-4:], **kwargs), 0, 0

    def forward(self, *xs, **kwargs):
        if len(xs) > 1:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)

class kp_group(nn.Module):
    def __init__(
        self, n, nstack, dims, modules, out_dim, pre=None, cnv_dim=256,
        make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
        make_regr_layer=make_kp_layer,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer,
        kp_layer=residual
    ):
        super(kp_group, self).__init__()
        print("Keypoint grouping enabled.")
        self.nstack    = nstack
        self._decode   = _decode_group
        curr_dim = dims[0]
        # 定义了预处理层，包括卷积层和残差层。
        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre
        # 定义了关键点模块的堆叠，每个模块包括上采样层、下采样层、hourglass层等。
        self.kps  = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            ) for _ in range(nstack)
        ])
        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])

        self.key_cnvs = nn.ModuleList([
            make_cnv_layer(cnv_dim, cnv_dim) for _ in range(nstack)
        ])
        self.center_cnvs = nn.ModuleList([
            make_cnv_layer(cnv_dim, cnv_dim) for _ in range(nstack)
        ])

        # keypoint heatmaps
        self.key_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])
        self.center_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        for key_heat, center_heat in zip(self.key_heats, self.center_heats):
            key_heat[-1].bias.data.fill_(-2.19)
            center_heat[-1].bias.data.fill_(-2.19)

        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])

        self.key_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.center_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])

        self.relu = nn.ReLU(inplace=True)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=260, nhead=4, dim_feedforward = 1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        self.classifier = nn.Sequential(
            nn.Linear(260, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )

    def _train(self, *xs):
        # 从输入元组 xs 中提取训练所需的参数，包括图像、关键点索引、中心索引、关键点长度和中心长度。
        image       = xs[0]
        key_inds    = xs[1]
        center_inds = xs[2]
        key_lens    = xs[3]
        center_lens = xs[4]
        # 接受图像作为输入并产生一个中间表示 inter
        inter = self.pre(image)
        outs  = []
        # 使用 zip 函数将多个网络层组合在一起，准备迭代
        layers = zip(
            self.kps, self.cnvs,
            self.key_cnvs, self.center_cnvs,
            self.key_heats, self.center_heats,
            self.key_regrs, self.center_regrs
        )
        for ind, layer in enumerate(layers):
            # 解包操作将每个组合层分解为不同的部分。
            kp_, cnv_          = layer[0:2]
            key_cnv_, center_cnv_   = layer[2:4]
            key_heat_, center_heat_ = layer[4:6]
            key_regr_, center_regr_ = layer[6:8]
            # 将中间表示 inter 通过不同的卷积层传递，并生成新的中间表示。
            kp  = kp_(inter)
            cnv = cnv_(kp)
            key_cnv = key_cnv_(cnv)
            center_cnv = center_cnv_(cnv)
            # 计算热图和回归
            key_heat, center_heat = key_heat_(key_cnv), center_heat_(center_cnv)
            key_regr, center_regr = key_regr_(key_cnv), center_regr_(center_cnv)
            #print(f"Shape of key_regrs {key_regrs.shape}")
            key_regr = _transpose_and_gather_feat(key_regr, key_inds)
            center_regr = _transpose_and_gather_feat(center_regr, center_inds)
            # 将计算的热图和回归值添加到输出列表中
            outs += [key_heat, center_heat, key_regr, center_regr]
            # 如果不是最后一个层，更新中间表示
            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        
        batch_size = key_cnv.shape[0]
        _, _, height, width = key_cnv.size() # heatmap size
        key_feat = _transpose_and_gather_feat(key_cnv, key_inds)
        center_feat = _transpose_and_gather_feat(center_cnv, center_inds)

        group_preds = []
        for b_ind in range(batch_size):
            # 过滤长度为零的批次
            if center_lens[b_ind] == 0 or key_lens[b_ind] == 0: continue
            # 限制中心长度的最大值。
            cen_len = min(center_lens[b_ind], 40)
            # 创建中心嵌入，其中包括位置和类型信息。
            cen_emb = center_feat[b_ind][:cen_len, :]
            tmp_inds = center_inds[b_ind][:cen_len].float()
            cen_pos = torch.stack([(tmp_inds % width) / width , (tmp_inds // width) / height]).transpose(0,1) # todo: add regrs
            cen_type = torch.ones((cen_pos.size(0),1)).float().cuda()
            cen_emb = torch.cat((cen_emb, cen_pos, cen_type), 1) 
            cen_type2 = torch.eye(cen_emb.size(0)).unsqueeze(-1).cuda()
            cen_emb = cen_emb.unsqueeze(1).repeat(1,cen_emb.size(0),1)
            cen_emb = torch.cat((cen_emb, cen_type2), -1) 
            # 创建关键点嵌入，与中心嵌入的创建过程相似。
            key_emb = key_feat[b_ind][:key_lens[b_ind], :]
            tmp_inds = key_inds[b_ind][:key_lens[b_ind]].float()
            key_pos = torch.stack([(tmp_inds % width) / width , (tmp_inds // width) / height]).transpose(0,1)
            key_type = torch.zeros((key_pos.size(0),2)).float().cuda()
            key_emb = torch.cat((key_emb, key_pos, key_type), 1)
            key_emb = key_emb.unsqueeze(1).repeat(1, cen_len, 1) # key_len * cen_len (batch_len) * featdim
            # 使用 encoder 对源数据进行变换，并通过分类器进行分类。最后的结果被添加到组预测列表中。
            src = torch.cat((cen_emb, key_emb), 0)
            out = self.transformer_encoder(src).transpose(1,0)
            out = self.classifier(out)
            group_preds.append(out.reshape(-1,2))
        
        return (outs, tuple(group_preds))

    def _test(self, *xs, **kwargs):
        image = xs[0]

        inter = self.pre(image)
        outs  = []

        layers = zip(
            self.kps, self.cnvs,
            self.key_cnvs, self.center_cnvs,
            self.key_heats, self.center_heats,
            self.key_regrs, self.center_regrs
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]
            key_cnv_, center_cnv_   = layer[2:4]
            key_heat_, center_heat_ = layer[4:6]
            key_regr_, center_regr_ = layer[6:8]

            kp  = kp_(inter)
            cnv = cnv_(kp)

            if ind == self.nstack - 1:
                key_cnv = key_cnv_(cnv)
                center_cnv = center_cnv_(cnv)

                key_heat, center_heat = key_heat_(key_cnv), center_heat_(center_cnv)
                key_regr, center_regr = key_regr_(key_cnv), center_regr_(center_cnv)

                outs += [key_heat, center_heat, key_regr, center_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)

        detections_key, detections_cen, key_inds, center_inds, key_scores, center_scores = self._decode(*outs[-4:], **kwargs)
        
        # grouping
        b_ind = 0 # assume batch = 1 during inference
        _, _, height, width = key_cnv.size() # heatmap size
        key_feat    = _transpose_and_gather_feat(key_cnv, key_inds)
        center_feat = _transpose_and_gather_feat(center_cnv, center_inds)
        key_len = (key_scores[b_ind] > 0.4).sum()
        cen_len = (center_scores[b_ind] > 0.4).sum()
        if key_len == 0 or cen_len == 0: return detections_key, detections_cen, torch.zeros((1,1))
        
        cen_emb = center_feat[b_ind][:cen_len, :]
        tmp_inds= center_inds[b_ind][:cen_len].float()
        cen_pos = torch.stack([(tmp_inds % width) / width , (tmp_inds // width) / height]).transpose(0,1)
        cen_type = torch.ones((cen_pos.size(0),1)).float().cuda()
        cen_emb = torch.cat((cen_emb, cen_pos, cen_type), 1) 
        cen_type2 = torch.eye(cen_emb.size(0)).unsqueeze(-1).cuda()
        cen_emb = cen_emb.unsqueeze(1).repeat(1,cen_emb.size(0),1)
        cen_emb = torch.cat((cen_emb, cen_type2), -1) 
        
        key_emb = key_feat[b_ind][:key_len, :]
        tmp_inds = key_inds[b_ind][:key_len].float()
        key_pos = torch.stack([(tmp_inds % width) / width , (tmp_inds // width) / height]).transpose(0,1)
        key_type = torch.zeros((key_pos.size(0),2)).float().cuda()
        key_pos = torch.cat((key_pos, key_type), 1)
        key_emb = torch.cat((key_emb, key_pos), 1)
        key_emb = key_emb.unsqueeze(1).repeat(1, cen_emb.size(1), 1) 

        src = torch.cat((cen_emb, key_emb), 0)
        out = self.transformer_encoder(src).transpose(1,0)
        out = self.classifier(out)
        out = nn.functional.softmax(out, dim = -1)
        out = out[:,:, 1]
        
        return detections_key, detections_cen, out

    def forward(self, *xs, **kwargs):
        if len(xs) > 1:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)

class DetectionLoss(nn.Module):
    def __init__(self, lambda_, lambda_b, regr_weight=1, focal_loss=_neg_loss):
        super(DetectionLoss, self).__init__()

        self.regr_weight = regr_weight
        self.focal_loss  = focal_loss
        self.regr_loss   = _regr_loss
        self.lambda_ = lambda_
        self.lambda_b = lambda_b

    def forward(self, outs, targets):
        stride = 4
        #print(f"Outs: {outs}")
        #print(f"Type of outs {type(outs)}")
        #for i, tensor in enumerate(outs):
            #print(f"The shape of tensor at outs index {i} is {tensor.shape}")
        #print(f"Shape of outs {get_shape(outs)}")
        
        key_heats = outs[0::stride]
        #for i, tensor in enumerate(key_heats):
            #print(f"The shape of tensor at key heats index {i} is {tensor.shape}")
        #print(f"Shape of key_heats {key_heats.shape}")
        center_heats = outs[1::stride]
        #print(f"Shape of center_heats {center_heats.shape}")
        key_regrs = outs[2::stride]
        #print(f"Shape of key_regrs {key_regrs.shape}")
        center_regrs = outs[3::stride]
        #print(f"Shape of center_regrs {center_regrs.shape}")
        gt_key_heat = targets[0]
        gt_center_heat = targets[1]
        gt_key_mask    = targets[2]
        gt_center_mask    = targets[3]
        gt_key_regr = targets[4]
        gt_center_regr = targets[5]

        # focal loss
        focal_loss = 0
        key_heats = [_sigmoid(t) for t in key_heats]
        center_heats = [_sigmoid(b) for b in center_heats]

        focal_loss += self.focal_loss(key_heats, gt_key_heat, self.lambda_, self.lambda_b) / 2
        focal_loss += self.focal_loss(center_heats, gt_center_heat, self.lambda_, self.lambda_b) 

        regr_loss = 0
        for key_regr, center_regr in zip(key_regrs, center_regrs):
            regr_loss += self.regr_loss(key_regr, gt_key_regr, gt_key_mask)
            regr_loss += self.regr_loss(center_regr, gt_center_regr, gt_center_mask)
        regr_loss = self.regr_weight * regr_loss

        loss = (focal_loss + regr_loss) / len(key_heats)
        return loss.unsqueeze(0)
    
class GroupingLoss(nn.Module):
    def __init__(self, lambda_, lambda_b, regr_weight=1, focal_loss=_neg_loss):
        super(GroupingLoss, self).__init__()

        self.regr_weight = regr_weight
        self.focal_loss  = focal_loss
        self.regr_loss   = _regr_loss
        self.lambda_ = lambda_
        self.lambda_b = lambda_b
        self.group_loss = nn.CrossEntropyLoss()
    # 定义了如何计算总损失。参数 outputs 包括预测的输出，而 targets 包括真实的目标值。
    def forward(self, outputs, targets):
        stride = 4
        
        outs, group_preds = outputs
        # 将输出切分成不同的部分。这里有关键点热图（key heats）、中心点热图（center heats）、关键点回归（key regrs）和中心点回归（center regrs）
        key_heats = outs[0::stride]
        center_heats = outs[1::stride]
        key_regrs = outs[2::stride]
        center_regrs = outs[3::stride]
        # 从目标中提取了各个部分，如关键点和中心点的真实热图，以及其他与目标有关的数据。
        gt_key_heat = targets[0]
        gt_center_heat = targets[1]
        gt_key_mask    = targets[2]
        gt_center_mask    = targets[3]
        gt_key_regr = targets[4]
        gt_center_regr = targets[5]
        group_targets = targets[6]
        tag_lens_center = targets[7]
        tag_lens_key = targets[8]

        # 首先，通过 S 型函数（sigmoid）激活预测的热图，然后使用预定义的 focal_loss 函数计算损失。
        focal_loss = 0
        key_heats = [_sigmoid(t) for t in key_heats]
        center_heats = [_sigmoid(b) for b in center_heats]

        focal_loss += self.focal_loss(key_heats, gt_key_heat, self.lambda_, self.lambda_b) / 2
        focal_loss += self.focal_loss(center_heats, gt_center_heat, self.lambda_, self.lambda_b) 
        # 计算了回归损失，包括关键点和中心点的回归。
        regr_loss = 0
        for key_regr, center_regr in zip(key_regrs, center_regrs):
            regr_loss += self.regr_loss(key_regr, gt_key_regr, gt_key_mask)
            regr_loss += self.regr_loss(center_regr, gt_center_regr, gt_center_mask)
        regr_loss = self.regr_weight * regr_loss
        
        # remove empty group targets
        group_targets_trim = []
        for b_ind in range(group_targets.size(0)):
            cen_len = min(tag_lens_center[b_ind], 40)
            tmp = group_targets[b_ind][:cen_len, :tag_lens_key[b_ind]]
            tmp = torch.cat((torch.zeros((cen_len,cen_len)).cuda().long(), tmp), 1)
            if tmp.reshape(-1).size(0) == 0: continue
            group_targets_trim.append(tmp.reshape(-1))
        
        group_loss = 0
        for b_ind in range(len(group_targets_trim)):
            group_loss += self.group_loss(group_preds[b_ind], group_targets_trim[b_ind])
        group_loss = 10 * group_loss # lr = 0.000025
        # print('focal_loss:', focal_loss.item(), 'regr_loss:', regr_loss.item(), 'group_loss:', group_loss.item())
        # 计算了总损失，将前面计算的三个损失组合在一起，并返回。
        if group_loss == 0:
            loss = (focal_loss + regr_loss) / len(key_heats)
        else:
            loss = (focal_loss + regr_loss + group_loss) / len(key_heats)
        return loss.unsqueeze(0)