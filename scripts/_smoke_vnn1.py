"""冒烟测试：8 个细胞验证 overexpress VNN1 + cls_and_gene 全链路。"""
import sys
import time
from pathlib import Path

ROOT = Path(r"E:\Geneformer")
sys.path.insert(0, str(ROOT / "tools"))

from geneformer import InSilicoPerturber
from geneformer_safe import vram_report

VNN1 = "ENSG00000112299"
OUT = ROOT / "outputs" / "_smoke_vnn1"

info = vram_report()
print(f"设备: {info['device']}  {info['total_gb']}G", flush=True)

isp = InSilicoPerturber(
    perturb_type="overexpress",
    perturb_rank_shift=None,
    genes_to_perturb=[VNN1],
    combos=0,
    anchor_gene=None,
    model_type="Pretrained",
    num_classes=0,
    emb_mode="cls_and_gene",
    cell_emb_style="mean_pool",
    filter_data=None,
    cell_states_to_model=None,
    state_embs_dict=None,
    max_ncells=8,
    cell_inds_to_perturb="all",
    emb_layer=0,
    forward_batch_size=4,
    nproc=8,
    model_version="V2",
)
t0 = time.time()
isp.perturb_data(str(ROOT / "models" / "V2-104M"),
                 str(ROOT / "data" / "tokenized" / "liver_tumor.dataset"),
                 str(OUT), "smoke")
print(f"OK  用时 {time.time()-t0:.0f}s", flush=True)

import pickle
for f in sorted(OUT.glob("*_raw.pickle")):
    d = pickle.load(open(f, "rb"))
    keys = list(d.keys())
    print(f"\n{f.name}: {len(keys)} keys")
    print("  示例 key:", keys[:3])
    v = d[keys[0]]
    print(f"  值个数: {len(v)}  前3: {v[:3]}")
