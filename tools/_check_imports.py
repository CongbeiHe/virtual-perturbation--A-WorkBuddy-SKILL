import pandas, numpy, scanpy, anndata, torch, transformers
print('pandas', pandas.__version__)
print('numpy ', numpy.__version__)
print('scanpy', scanpy.__version__)
print('anndata', anndata.__version__)
print('torch', torch.__version__)
print('transformers', transformers.__version__)
from geneformer import EmbExtractor, TranscriptomeTokenizer, Classifier
print('geneformer package importable: EmbExtractor/Tokenizer/Classifier all OK')