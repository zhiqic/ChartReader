import numpy as np
from db.base import BASE

class DETECTION(BASE):
    def __init__(self, db_config):
        super().__init__()

        self._configs["categories"]      = 8
        self._configs["input_size"]      = [511]
        self._configs["output_sizes"]    = [[128, 128]]
        self._configs["lighting"] = True
        self._configs["nms_threshold"]   = 0.5
        self._configs["max_per_image"]   = 100
        self._configs["top_k"]           = 100
        self._configs["nms_kernel"]      = 3

        self._configs["gaussian_bump"]   = True
        self._configs["gaussian_iou"]    = 0.7
        self._configs["gaussian_radius"] = -1

        self._configs["rand_color"] = False

        self.update_config(db_config)