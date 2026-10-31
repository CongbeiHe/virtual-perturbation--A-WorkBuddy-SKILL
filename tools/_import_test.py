"""逐步导入诊断，定位崩溃点。"""

import sys

print("python", sys.version.split()[0], flush=True)

try:
    import numpy

    print("numpy      ", numpy.__version__, flush=True)
except Exception as exc:
    print("numpy FAIL", exc, flush=True)

try:
    import pandas

    print("pandas     ", pandas.__version__, flush=True)
except Exception as exc:
    print("pandas FAIL", exc, flush=True)

try:
    import torch

    print("torch      ", torch.__version__, "cuda:", torch.cuda.is_available(), flush=True)
except Exception as exc:
    print("torch FAIL", exc, flush=True)

try:
    import transformers

    print("transformers", transformers.__version__, flush=True)
except Exception as exc:
    print("transformers FAIL", type(exc).__name__, exc, flush=True)

try:
    import datasets

    print("datasets   ", datasets.__version__, flush=True)
except Exception as exc:
    print("datasets FAIL", type(exc).__name__, exc, flush=True)

try:
    import geneformer

    print("geneformer  OK ->", geneformer.__file__, flush=True)
except Exception as exc:
    import traceback

    print("geneformer FAIL", type(exc).__name__, exc, flush=True)
    traceback.print_exc()
