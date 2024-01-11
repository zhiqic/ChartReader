import os
import numpy as np

class BASE():
    def __init__(self):
        self._split = None
        self._db_inds = []
        self._image_ids = []

        self._data            = "chart"
        self._image_file      = None
        # 存储图像数据的均值。通常用于归一化操作。设置为三个浮点数的零数组，可能代表RGB通道的均值。
         # 存储特征值
        self._eig_val = np.ones((3, ), dtype=np.float32)
        # 存储特征向量
        self._eig_vec = np.zeros((3, 3), dtype=np.float32)

        self._mean = np.zeros((3, ), dtype=np.float32)
        # 存储图像数据的标准差。通常用于归一化操作。设置为三个浮点数的一数组，可能代表RGB通道的标准差。
        self._std = np.ones((3, ), dtype=np.float32)

        self._configs = {}

        self._data_rng = None

    @property
    def data(self):
        if self._data is None:
            raise ValueError("data is not set")
        return self._data

    @property
    def configs(self):
        return self._configs

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    @property
    def eig_val(self):
        return self._eig_val

    @property
    def eig_vec(self):
        return self._eig_vec

    @property
    def db_inds(self):
        return self._db_inds

    @property
    def split(self):
        return self._split

    def update_config(self, new):
        for key in new:
            if key in self._configs:
                self._configs[key] = new[key]

    def image_ids(self, ind):
        return self._image_ids[ind]

    def image_file(self, ind):
        if self._image_file is None:
            raise ValueError("Image path is not initialized")
        # 从 self._image_ids 列表中获取索引为 ind 的图像ID
        image_id = self._image_ids[ind]
        # 使用 Python 的字符串格式化功能 .format() 来将图像ID插入到 self._image_file 字符串中。例如，如果 self._image_file 是一个字符串模板，如 "path/to/image/{}.jpg"，那么 .format(image_id) 会用 image_id 替换 {}，从而生成完整的文件路径。
        return self._image_file.format(image_id)

    def evaluate(self, name):
        pass

    def shuffle_inds(self, quiet=False):
        if self._data_rng is None:
            self._data_rng = np.random.RandomState(os.getpid())

        if not quiet:
            print("Shuffling...")
        rand_perm = self._data_rng.permutation(len(self._db_inds))
        self._db_inds = self._db_inds[rand_perm]
