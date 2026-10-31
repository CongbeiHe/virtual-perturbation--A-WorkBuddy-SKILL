"""步骤二：tokenize -> InSilicoPerturber -> 统计表。

移植自 scripts/22，方法学常量见 config.py，全部与已验证实验一致：
  - TranscriptomeTokenizer(special_token=True, use_h5ad_index=True, model_version=V2)
  - InSilicoPerturber(emb_mode="cls_and_gene", emb_layer=0, fp32, batch=8)
  - InSilicoPerturberStats(aggregate_gene_shifts + aggregate_data)
  - raw pickle 自动分发到 gene/ 与 cell/ 目录
  - 各步骤幂等：输出已存在则跳过
"""
import pickle
import shutil
import time
from pathlib import Path

from .config import (BATCH, CODE_DIR, EMB_LAYER, EMB_MODE, CELL_EMB_STYLE,
                     MODEL_DIR, MODEL_VERSION, NPROC, PERTURB_MODES)


def log(m=""):
    print(m, flush=True)


def banner(title):
    log()
    log("=" * 70)
    log(f"  {title}")
    log("=" * 70)


def _ensembl_of(symbol_or_ensembl: str) -> str:
    """符号或 Ensembl 均可；返回 Ensembl ID。"""
    if symbol_or_ensembl.startswith("ENSG"):
        return symbol_or_ensembl
    g2i = pickle.load(open(CODE_DIR / "gene_name_id_dict_gc104M.pkl", "rb"))
    eid = g2i.get(symbol_or_ensembl)
    assert eid, f"基因 {symbol_or_ensembl} 无法映射到 gc104M Ensembl"
    return eid


def run_isp(dataset_name, target_gene, mode="overexpress",
            counts_dir=None, dataset_path=None, dirs=None, batch=BATCH):
    """完整 ISP 主流水线，返回 dict(输出路径)。

    幂等：tokenized dataset / raw pickle / stats csv 已存在时自动跳过。
    """
    from .config import default_dirs

    if dirs is None:
        dirs = default_dirs(dataset_name, target_gene, mode)
    ds_path = Path(dataset_path) if dataset_path else dirs["dataset"]
    perturb_type = PERTURB_MODES[mode]
    target_eid = _ensembl_of(target_gene)

    # ---- tokenize
    from geneformer import TranscriptomeTokenizer

    if ds_path.exists():
        log(f"[skip] {ds_path} 已存在")
    else:
        banner(f"步骤 1：tokenize {dataset_name}")
        src_dir = Path(counts_dir) if counts_dir else dirs["counts_dir"]
        tk = TranscriptomeTokenizer(
            custom_attr_name_dict={"cell_type": "cell_type", "SampleID": "SampleID",
                                   "Patient": "Patient", "Tissue": "Tissue",
                                   "Condition": "Condition", "malignant": "malignant"},
            nproc=NPROC, model_version=MODEL_VERSION,
            use_h5ad_index=True, special_token=True,
        )
        t0 = time.time()
        tk.tokenize_data(str(src_dir), str(ds_path.parent), dataset_name, file_format="h5ad")
        log(f"tokenize 完成，用时 {time.time()-t0:.0f}s")

    # ---- 体检
    import numpy as np
    from datasets import load_from_disk

    banner("步骤 2：tokenized 数据体检")
    tok = pickle.load(open(CODE_DIR / "token_dictionary_gc104M.pkl", "rb"))
    cls_id, eos_id = tok["<cls>"], tok["<eos>"]
    tgt_tok = tok[target_eid]
    log(f"<cls>={cls_id}  <eos>={eos_id}  {target_gene} token={tgt_tok}")
    ds = load_from_disk(str(ds_path))
    ids = ds["input_ids"]
    n = len(ds)
    lens = [len(x) for x in ids]
    n_cls = sum(1 for x in ids if x[0] == cls_id)
    n_eos = sum(1 for x in ids if x[-1] == eos_id)
    n_tgt = sum(1 for x in ids if tgt_tok in x)
    log(f"细胞 {n}  长度 中位={float(np.median(lens)):.0f} 最大={max(lens)}  "
        f"<cls>开头 {n_cls}  <eos>结尾 {n_eos}  含{target_gene} {n_tgt} ({n_tgt/n:.1%})")
    assert n_cls == n and n_eos == n, "特殊 token 缺失！"

    # ---- ISP
    from geneformer import InSilicoPerturber
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    from geneformer_safe import vram_report

    raw_dir = Path(dirs["isp_raw"])
    for d in [raw_dir, dirs["isp_gene"], dirs["isp_cell"]]:
        Path(d).mkdir(parents=True, exist_ok=True)

    if any(raw_dir.glob("*_raw.pickle")):
        log("[skip] raw pickle 已存在，跳过扰动计算")
    else:
        banner(f"步骤 3：{mode} {target_gene}（fp32, {EMB_MODE}, batch={batch}）")
        info = vram_report()
        log(f"设备: {info['device']}  总显存 {info['total_gb']}G")
        isp = InSilicoPerturber(
            perturb_type=perturb_type,
            perturb_rank_shift=None,
            genes_to_perturb=[target_eid],
            combos=0,
            anchor_gene=None,
            model_type="Pretrained",
            num_classes=0,
            emb_mode=EMB_MODE,
            cell_emb_style=CELL_EMB_STYLE,
            filter_data=None,
            cell_states_to_model=None,
            state_embs_dict=None,
            max_ncells=None,
            cell_inds_to_perturb="all",
            emb_layer=EMB_LAYER,
            forward_batch_size=batch,
            nproc=NPROC,
            model_version=MODEL_VERSION,
        )
        t0 = time.time()
        isp.perturb_data(str(MODEL_DIR), str(ds_path), str(raw_dir), dirs["tag"])
        log(f"扰动完成，用时 {(time.time()-t0)/60:.1f} 分钟")

    # ---- stats
    from geneformer import InSilicoPerturberStats

    banner("步骤 4：统计表")
    for f in raw_dir.glob("*_raw.pickle"):
        dst = dirs["isp_gene"] if "_gene_embs_dict_" in f.name else dirs["isp_cell"]
        if not (Path(dst) / f.name).exists():
            shutil.copy2(f, Path(dst) / f.name)

    stats_dir = Path(dirs["stats"])
    stats_dir.mkdir(parents=True, exist_ok=True)
    gene_csv = stats_dir / f"{dirs['tag']}_gene_shifts.csv"
    cell_csv = stats_dir / f"{dirs['tag']}_cell_shift.csv"

    if not gene_csv.exists():
        st = InSilicoPerturberStats(
            mode="aggregate_gene_shifts", genes_perturbed=[target_eid], combos=0,
            anchor_gene=None, cell_states_to_model=None,
            pickle_suffix="_raw.pickle", model_version=MODEL_VERSION,
        )
        st.get_stats(str(dirs["isp_gene"]), None, str(stats_dir), f"{dirs['tag']}_gene_shifts")
    if not cell_csv.exists():
        st2 = InSilicoPerturberStats(
            mode="aggregate_data", genes_perturbed=[target_eid], combos=0,
            anchor_gene=None, cell_states_to_model=None,
            pickle_suffix="_raw.pickle", model_version=MODEL_VERSION,
        )
        st2.get_stats(str(dirs["isp_cell"]), None, str(stats_dir), f"{dirs['tag']}_cell_shift")

    # ---- 预览
    import pandas as pd
    banner("步骤 5：结果预览")
    g = pd.read_csv(gene_csv)
    cols = [c for c in ["Perturbed", "Affected_gene_name", "Affected_Ensembl_ID",
                        "Cosine_sim_mean", "Cosine_sim_stdev", "N_Detections"] if c in g.columns]
    log(f"下游基因表 {g.shape[0]} 行；位移最大的 20 个：")
    log(g.nsmallest(20, "Cosine_sim_mean")[cols].to_string(index=False))
    c = pd.read_csv(cell_csv)
    log(f"\n细胞位移表 {c.shape[0]} 行")

    return {"gene_shifts": gene_csv, "cell_shift": cell_csv,
            "raw_dir": raw_dir, "n_cells": n,
            "n_express_target": n_tgt}
