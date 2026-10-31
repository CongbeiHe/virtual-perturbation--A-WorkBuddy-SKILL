"""查看 tokenized .dataset 的基本统计。"""

import sys
from pathlib import Path

from datasets import load_from_disk
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else r"E:\Geneformer\data\tokenized\pbmc3k.dataset"
ds = load_from_disk(path)
print("路径   :", path)
print("细胞数 :", len(ds))
print("字段   :", ds.column_names)
print("特征   :", ds.features if hasattr(ds, "features") else "-")

lengths = np.array([len(x) for x in ds["input_ids"]])
print("序列长度: min=%d  median=%d  mean=%.1f  max=%d" % (
    lengths.min(), np.median(lengths), lengths.mean(), lengths.max()))

first = ds[0]["input_ids"]
print("首细胞前 20 个 token:", first[:20])
print("前 3 个特殊 token（应含 <cls>=2）:", [t for t in first[:3]])

for key in ds.column_names:
    if key != "input_ids":
        vals = ds[key][:5]
        print(f"标签 {key}: {set(vals) if isinstance(vals[0], str) else vals[:5]}")
