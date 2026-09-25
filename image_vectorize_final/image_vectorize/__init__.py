from .image_vectorize import image_vectorize

# 向量距离计算方法：欧氏距离。
# 特征各段均已 L2 归一化，欧氏距离与余弦距离单调等价，且计算更省。
METRIC = "euclidean"

# 判定为"同一页"的距离阈值。
# 在 test_samples 两组用例上实测标定：
# - 同页（model/real 同名）对距离最大值约 1.26；
# - 新页（model 中不存在）与所有 model 距离的最小值约 1.50。
# 取区间中点 1.38，两侧各留约 0.12 的安全边距。
DIFFERENCE_THRESHOLD = 1.38

__all__ = ["image_vectorize", "METRIC", "DIFFERENCE_THRESHOLD"]
