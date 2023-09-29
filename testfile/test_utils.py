import numpy as np
# 根据给定的比例、边界和尺寸重新缩放检测框中的点
# dets: 检测框，通常是一个三维数组，其中每个检测框包括位置信息。
# ratios: 重新缩放的比例，用于调整检测框的大小。
# borders: 边界偏移量，用于调整检测框的位置。
# sizes: 新的尺寸范围，用于限制检测框的大小。
def _rescale_points(dets, ratios, borders, sizes):
    # 从检测框中提取x和y坐标。
    xs, ys = dets[:, :, 2], dets[:, :, 3]
    # 通过除以给定的x方向比例来重新缩放x坐标。
    xs    /= ratios[0, 1]
    # 通过除以给定的y方向比例来重新缩放y坐标。
    ys    /= ratios[0, 0]
    # 将x, y坐标减去边界偏移量，以调整位置。
    xs    -= borders[0, 2]
    ys    -= borders[0, 0]
    # 使用numpy的clip函数将x, y坐标限制在新的尺寸范围内。
    # 如果数组中的元素低于下限，则将其设置为下限值；如果高于上限，则将其设置为上限值。对于在下限和上限之间的值，np.clip 会保留原始值。
    np.clip(xs, 0, sizes[0, 1], out=xs)
    np.clip(ys, 0, sizes[0, 0], out=ys)