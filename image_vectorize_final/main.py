from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import List, Sequence, Tuple

import numpy as np
from sklearn.metrics import pairwise_distances


try:
    from image_vectorize import image_vectorize, METRIC, DIFFERENCE_THRESHOLD
except ImportError:
    from demo import image_vectorize, METRIC, DIFFERENCE_THRESHOLD
    print("\033[31m没有在 image_vectorize 中找到 image_vectorize，使用 demo。\033[0m")


TEST_SAMPLE_DIR: Path = Path("./test_samples").resolve()

# 平均单次向量化不能超过这个时间（ms）
VECTORIZATION_AVG_TIME_LIMIT: float = 100.0


@dataclass
class Page:
    number: str
    vector: np.ndarray


def vectorize_sample_dir(dir_path: Path) -> List[Page]:
    """
    将一个文件夹里的图片进行向量化。
    """
    print(f"向量化 {dir_path.parts[-1]} ：")

    # 读取图片
    numbers: List[str] = []
    image_bytes: List[bytes] = []
    image_count = 0
    for file_name in os.listdir(dir_path):
        number, suffix = os.path.splitext(file_name)
        if suffix not in (".jpg", ".png"):
            continue

        numbers.append(number)
        file_path = dir_path / file_name
        with open(file_path, mode="rb") as file:
            image_bytes.append(file.read())
        image_count += 1

    if image_count == 0:
        return []

    # 图片向量化
    print("各图片的向量化耗时（ms）：")
    print("".join(number.rjust(8) for number in numbers))
    total_time = 0
    image_vectors: List[np.ndarray] = []
    for i in range(image_count):
        start_time = time.time()
        image_vector = image_vectorize(image_bytes[i])
        end_time = time.time()
        image_vectors.append(image_vector)
        time_consuming = (end_time - start_time) * 1000
        total_time += time_consuming
        print(f"{time_consuming:.1f}".rjust(8), end="")

    avg_time = total_time / image_count
    color = "\033[32m"
    if avg_time > VECTORIZATION_AVG_TIME_LIMIT:
        color = "\033[31m"
    print(f"\n平均耗时（ms）：\n\t{color}{avg_time:.1f}\033[0m")

    return [
        Page(numbers[i], image_vectors[i])
        for i in range(image_count)
    ]


def vector_comparison(model_pages: Sequence[Page], real_pages: Sequence[Page]):
    """
    向量之间的比较。
    """
    model_numbers = [page.number for page in model_pages]
    print("距离：")
    print("model numebrs:" + "".join(number.rjust(8) for number in model_numbers))
    print("real numbers ↓")
    matches: List[Tuple[Page, Page]] = []
    for real_page in real_pages:
        print(real_page.number.rjust(14), end="")
        min_dist = DIFFERENCE_THRESHOLD
        match = None
        for model_page in model_pages:
            dist = pairwise_distances(
                real_page.vector.reshape(1, -1),
                model_page.vector.reshape(1, -1),
                metric=METRIC
            )[0, 0]
            if dist < min_dist and dist < DIFFERENCE_THRESHOLD:
                min_dist = dist
                match = model_page

            # 判断正误
            correct = None
            if dist < DIFFERENCE_THRESHOLD:
                correct = real_page.number == model_page.number
            elif real_page.number == model_page.number:
                correct = False

            color = "\033[0m"
            if correct:
                color = "\033[32m"
            elif correct is False:
                color = "\033[31m"
            print(color + f"{dist:.2f}".rjust(8) +"\033[0m", end="")
        matches.append((real_page, match))
        print()

    print("\n匹配：")
    print(" real -> model")
    correct_count = 0
    for real_page, match in matches:
        # 判断正误
        correct = match is None
        expect = None
        if real_page.number in model_numbers:
            correct = match and real_page.number == match.number
            expect = real_page.number
        correct_count += bool(correct)

        # 打印匹配结果
        print("\033[32m" if correct else "\033[31m", end="")
        print(real_page.number.rjust(5), "->", end=" ")
        if match:
            print(match.number, end="")
        else:
            print("None", end="")
        if not correct:
            print(f"\texpect: {expect}", end="")
        print("\033[0m")

    print(f"正确率：{correct_count / len(real_pages) * 100:.1f}%")


def test_sample(sample_name: str):
    """
    测试一个样例。
    """
    print(f"\n==== Test {sample_name} ====\n")
    sample_dir = TEST_SAMPLE_DIR / sample_name
    model_dir = sample_dir / "model"
    real_dir = sample_dir / "real"

    # 图片向量化
    model_pages = vectorize_sample_dir(model_dir)
    print()
    real_pages = vectorize_sample_dir(real_dir)

    # 向量比对
    print()
    vector_comparison(model_pages, real_pages)


def main():
    for sample_name in os.listdir(TEST_SAMPLE_DIR):
        test_sample(sample_name)


if __name__ == '__main__':
    main()
