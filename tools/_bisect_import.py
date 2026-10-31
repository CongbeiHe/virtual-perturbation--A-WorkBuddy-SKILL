"""逐个子模块导入，定位崩溃点。每个模块独立进程，避免崩溃互相影响。"""

import subprocess
import sys

MODULES = [
    "geneformer.tokenizer",
    "geneformer.collator_for_classification",
    "geneformer.emb_extractor",
    "geneformer.in_silico_perturber",
    "geneformer.in_silico_perturber_stats",
    "geneformer.pretrainer",
    "geneformer.classifier",
    "geneformer.mtl_classifier",
    "geneformer.mtl.utils",
]

for mod in MODULES:
    code = f"import {mod}; print('ok')"
    p = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    tail = (p.stderr or "").strip().splitlines()
    tail = tail[-1] if tail else ""
    status = "OK  " if p.returncode == 0 else f"FAIL({p.returncode:#x})"
    print(f"{status:>12}  {mod:<45} {tail[:150]}")
