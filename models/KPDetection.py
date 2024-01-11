from .py_utils import kp_detection, DetectionLoss, _neg_loss, residual
from .model_utils import make_hg_layer, make_pool_layer

class Model(kp_detection):
    def __init__(self):
        n = 5
        dims = [256, 256, 384, 384, 384, 512]
        modules = [2, 2, 2, 2, 2, 4]
        out_dim = 3

        super(Model, self).__init__(
            n, 2, dims, modules, out_dim,
            make_pool_layer=make_pool_layer,
            make_hg_layer=make_hg_layer,
            kp_layer=residual, cnv_dim=256
        )

loss = DetectionLoss(focal_loss=_neg_loss, lambda_=4, lambda_b=2)