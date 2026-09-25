# 图片向量化

给定一张书页展平后的图片，输出一个 1344 维特征向量：内容相同的图片
向量距离小，不同页面距离大。用于绘本阅读场景中"客户端上传页 ↔ 服务端
缓存页"的快速匹配——上传图只需向量化一次，即可与全部缓存页向量做
距离比较，无需跑任何大模型。

## 指标速览

| 指标 | 结果 | 要求 |
| --- | --- | --- |
| 哆啦A梦 正确率 | **100.0%**（10/10） | — |
| 波西和皮普 正确率 | **100.0%**（16/16） | — |
| 单张平均耗时 | 7.1 ~ 15.4 ms | ≤ 100 ms |
| 向量维度 | 1344（float64） | 一维，整型或浮点型 |
| 合成退化压力测试 | 279/308（11 类退化） | 通用性自检 |
| 依赖 | numpy、Pillow | 允许第三方库，不调用大模型 |

完整评测输出见 [`results/main运行结果.txt`](results/main运行结果.txt)；
压力测试与边界验证输出见 [`results/压力测试与边界验证.txt`](results/压力测试与边界验证.txt)。

## 调用方式

```python
from image_vectorize import image_vectorize, METRIC, DIFFERENCE_THRESHOLD

with open("image_1.png", mode="rb") as file:
    image_bytes = file.read()

image_vector = image_vectorize(image_bytes)   # np.ndarray, shape=(1344,), float64
```

本包导出的三个对象：

| 名称 | 值 | 说明 |
| --- | --- | --- |
| `image_vectorize` | 函数 | 图片字节 → 特征向量 |
| `METRIC` | `"euclidean"` | 向量距离计算方法（sklearn `pairwise_distances` 可用） |
| `DIFFERENCE_THRESHOLD` | `1.38` | 距离小于该值判定为同一页；标定过程见 [实验与阈值分析](docs/实验与阈值分析.md) |

异常约定：空字节或非 bytes 类型输入抛出带说明的 `ValueError`；字节流
无法解码时抛出 `PIL.UnidentifiedImageError`。

## 算法概述

结构 + 颜色两段互补特征，拼接后用欧氏距离比较：

1. **解码**：Pillow 打开字节流；JPEG 走 `draft()` 路径在 DCT 域直接
   降采样解码，1600×1100 的页面解码仅数毫秒。
2. **结构通道（576 维）**：64×64 灰度网格 → 高光压缩（截断至 120/255
   后线性拉伸）→ 局部照度归一化（float64 积分图求 7×7 邻域均值后作除）
   → 全局标准化 → 二维 DCT-II 取 24×24 低频 → L2 归一化。
3. **颜色通道（768 维）**：16×16 缩略图逐像素色度归一化 `rgb/(r+g+b)`
   → 三个平面各自标准化 → L2 归一化。
4. **拼接**：结构段权重 1.0，颜色段权重 0.6。

这套组合针对的是翻拍图的真实退化：重新压缩、轻微透视/弯曲形变、边缘
裁切差异、整页反光过曝、亮度对比度偏移。每一步的设计理由见
[算法详解](docs/算法详解.md)。

## 文件结构

```
final/
├── README.md                  # 本文件
├── image_vectorize/           # 交付代码包（评测程序从此导入）
│   ├── __init__.py            #   导出 image_vectorize / METRIC / DIFFERENCE_THRESHOLD
│   └── image_vectorize.py     #   向量化实现（含详细注释与自测入口）
├── main.py                    # 官方评测脚本副本（未改动）
├── requirements.txt           # 依赖清单（与上级目录一致）
├── docs/
│   ├── 算法详解.md            # 每一步的原理与设计理由
│   └── 实验与阈值分析.md      # 消融数据、退化压力测试、阈值标定
└── results/
    ├── main运行结果.txt       # 完整评测输出（含距离矩阵与逐张耗时）
    └── 压力测试与边界验证.txt # 合成退化压力测试 + 边界/异常输入验证
```

## 复现方法

```bash
cd image-vectorize-test          # 仓库根目录（含 test_samples/）
python main.py                   # 自动导入 image_vectorize 包
```

本目录内的 `main.py` 为官方脚本副本；若要在 `final/` 内直接运行，把仓库
根目录的 `test_samples/` 复制到 `final/` 下即可（脚本读取 `./test_samples`）。

环境：Python 3.12+（开发时用 3.13 验证），`python -m pip install -r requirements.txt`。

## 性能数据

`main.py` 实测（Windows，单核计时）：

| 样例组 | 图片集 | 张数 | 平均单张耗时 |
| --- | --- | --- | --- |
| 哆啦A梦 | model | 11 | 15.4 ms |
| 哆啦A梦 | real | 10 | 9.1 ms |
| 波西和皮普 | model | 17 | 7.1 ms |
| 波西和皮普 | real | 16 | 9.2 ms |

说明：

- 每组第一张约 60~70 ms，属首次调用的冷启动开销（磁盘缓存与库初始化），
  计入平均后仍远低于 100 ms 限值。
- PNG 输入不走 JPEG draft 加速路径，实测约 60~80 ms，同样满足限值。

## 通用性验证

**合成退化压力测试**：把全部 28 张 model 图各退化成 11 类比真实上传图
更狠的"假 real"（亮度 ±30%、对比度 −40%、gamma 0.6/1.6、JPEG 重压缩
q30、旋转 2°/4°、高斯噪声、椭圆反光斑、内容裁切 12%），要求退化图仍
检索回原图且距离低于阈值。9/11 类退化通过率 100%，旋转 4° 为 96.4%
（1 例压线），内容裁切 12% 不在本任务输入设定内（详见分析文档）。

**边界与异常输入**：PNG 大图、16px 极小图、灰度、RGBA、CMYK JPEG、
单色图、3000px 超大图均正常输出 1344 维有限向量；空字节、None、int
均抛出带信息的 `ValueError`。
