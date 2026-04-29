# KG-Enhanced LLM — Project Guide

## Purpose

A unified research framework for reproducing and comparing KG-enhanced LLM baselines on KGQA benchmarks. All baselines are evaluated on **WebQSP** (Freebase-grounded) with consistent data loading, metrics, and project structure.

## Project Structure

```
data_loader/
    base_webqsp_dataset.py   # Base PyTorch Dataset — all baselines inherit from this
    data_loaders.py          # WebQSPDataLoader (BaseDataLoader wrapper)
baselines/
    <MethodName>/
        __init__.py
        dataset.py           # Inherits WebQSPDataset, adds method-specific fields
        model.py             # Model architecture (to be implemented per baseline)
datasets/
    WebQSP/
        WebQSP.train.json    # 3098 questions
        WebQSP.test.json     # 1639 questions
utils/
    metric.py                # KGQA-specific Hits@1 and F1 (set-based, not classification)
```

## Baseline Overview

| Category | Methods | Extra dataset field(s) |
|----------|---------|----------------------|
| Embedding-based | KVMem, EmbedKGQA, NSM, TransferNet, KGT5 | `memories` / `entity_candidates` / `hop_entities` / `target_text` |
| Retrieval-based | GraftNet, PullNet, SR | `subgraph: {entities, tuples, passages}` |
| LLM-only | FlanT5, Alpaca, LLaMA2, ChatGPT | `prompt` (model-specific template) |
| LLMs+KGs | KDCoT, UniKGQA, ToG, KGCoT, RoG, SubgraphRAG | `triples` / `subgraph` / `beam_paths` / `reasoning_paths` / `scored_triples` |
| Proposed | ReliableReasoningPath | `semantic_paths` + `structural_paths` |

## Adding a New Baseline

1. Create `baselines/<Name>/dataset.py` inheriting `WebQSPDataset`:

```python
import json
from data_loader.base_webqsp_dataset import WebQSPDataset

class <Name>Dataset(WebQSPDataset):
    def __init__(self, data_path, preprocessed_path=None):
        super().__init__(data_path)
        for sample in self.samples:
            sample['<field>'] = <default>
        if preprocessed_path:
            self._merge(preprocessed_path)

    def _merge(self, path):
        with open(path) as f:
            index = {item['id']: item.get('<key>', <default>) for item in json.load(f)}
        for sample in self.samples:
            sample['<field>'] = index.get(sample['id'], <default>)
```

2. The preprocessed file is always indexed by `item['id']` (WebQSP question ID, e.g. `"WebQTrn-0"`).
3. Use `super().__init__()` first, then merge — do **not** override `_parse()` for external data.

## Base Dataset Fields

Every sample from `WebQSPDataset.__getitem__` contains:

```python
{
    'id':            str,           # e.g. "WebQTrn-0"
    'question':      str,           # ProcessedQuestion
    'topic_entity':  {'mid': str, 'name': str},
    'answers':       [{'mid': str, 'name': str}, ...],  # union across all parses
    'relation_path': [str, ...],    # InferentialChain from first valid parse
    'sparql':        str,
}
```

## Metrics

Metrics in `utils/metric.py` are **set-based**, not classification:

```python
hits_at_1(pred: str, golds: list[str]) -> float   # is top-1 pred in gold set?
f1_score(preds: list[str], golds: list[str]) -> float  # set precision/recall F1
```

Typical eval loop:
```python
golds = [a['mid'] for a in sample['answers']]
h1 = hits_at_1(top1_pred, golds)
f1 = f1_score(all_preds, golds)
```

## DataLoader

```python
from data_loader.data_loaders import WebQSPDataLoader

loader = WebQSPDataLoader(data_path, batch_size=32)
# batch is dict-of-lists: {'question': [...], 'answers': [...], ...}
```

`webqsp_collate` returns `dict[str, list]` — variable-length fields (answers, paths) stay as Python lists, not tensors.

## Environment

Managed via conda: `conda env create -f env.yaml && conda activate KG`

Key packages: PyTorch 2.11, CUDA 13.0. Additional packages required for LLM baselines (`transformers`, `peft`, `accelerate`, `sentencepiece`, `bitsandbytes`, `openai`, `sentence-transformers`, `faiss-cpu`, `datasets`, `einops`, `scikit-learn`) are declared in `env.yaml` but **must be installed separately via pip** if not present.

## Key Conventions

- **No image/vision imports** — `torchvision`, `transforms`, `MnistDataLoader` are not used.
- **Metrics take strings, not tensors** — pass entity MID strings or entity name strings to `hits_at_1` / `f1_score`.
- **Preprocessing is offline** — dataset classes load pre-extracted files; no runtime KG queries during training.
- **`_EMPTY_SUBGRAPH` pattern** — retrieval-based datasets define a module-level constant for the default empty subgraph to avoid allocating new dicts on every `.get()` miss.
- **LLM-only baselines use `prompt` field** — formatted in the dataset, not in the model/trainer.
- **RRP has two path fields** — `semantic_paths` (from LLM) and `structural_paths` (from relation embedding module), kept separate to match the two-module architecture in the paper.
