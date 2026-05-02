# KG-Enhanced LLM — Project Guide

## Purpose

A unified research framework for reproducing and comparing KG-enhanced LLM baselines on KGQA benchmarks. All baselines are evaluated on **WebQSP** (Freebase-grounded) with consistent data loading, metrics, and project structure.

## Project Structure

```
main.py                      # CLI entry point — train / eval all baselines
base/
    base_trainer.py          # Abstract BaseTrainer (epochs, checkpointing, resume)
    base_data_loader.py      # BaseDataLoader (train/val split via SubsetRandomSampler)
trainer/
    trainer.py               # Trainer(BaseTrainer) — loss, eval loop, lr scheduler
data_loader/
    base_webqsp_dataset.py   # Base PyTorch Dataset — all baselines inherit from this
    data_loaders.py          # webqsp_collate + WebQSPDataLoader (base dataset only)
baselines/
    <MethodName>/
        __init__.py
        dataset.py           # Inherits WebQSPDataset, adds method-specific fields
        model.py             # Model architecture
datasets/
    WebQSP/
        WebQSP.train.json    # 3098 questions
        WebQSP.test.json     # 1639 questions
utils/
    metric.py                # KGQA-specific Hits@1 and F1 (set-based, not classification)
    util.py                  # prepare_device(n_gpu_use) -> (device, gpu_list)
```

## Baseline Overview

| Category | Methods | Extra dataset field(s) |
|----------|---------|----------------------|
| Embedding-based | KVMem, EmbedKGQA, NSM, TransferNet, KGT5 | `memories` / `entity_candidates` / `hop_entities` / `target_text` |
| Retrieval-based | GraftNet, PullNet, SR | `subgraph: {entities, tuples, passages}` |
| LLM-only | FlanT5, Alpaca, LLaMA2, ChatGPT | `prompt` (model-specific template) |
| LLMs+KGs | KDCoT, UniKGQA, ToG, KGCoT, RoG, SubgraphRAG | `triples` / `subgraph` / `beam_paths` / `reasoning_paths` / `scored_triples` |
| Proposed | ReliableReasoningPath | `semantic_paths` + `structural_paths` |

## CLI — Running Experiments

All baselines are launched through `main.py`:

```sh
# Train
python main.py train --baseline <Name> [options]

# Eval
python main.py eval --baseline <Name> --checkpoint <path> [options]
```

Key flags:

| Flag | Default | Notes |
|------|---------|-------|
| `--baseline` | — | Required. Exact name from `REGISTRY` in `main.py`. |
| `--train-data` | `datasets/WebQSP/WebQSP.train.json` | |
| `--test-data` | `datasets/WebQSP/WebQSP.test.json` | |
| `--preprocessed` | `None` | Single pre-extracted JSON for memories / paths / triples (KVMem, NSM, KDCoT, RRP, …). |
| `--subgraph` | `None` | Subgraph JSON for retrieval-based baselines (GraftNet, PullNet, SR, UniKGQA, SubgraphRAG). |
| `--model-name` | model default | HuggingFace model name/path for LLM-based models. |
| `--epochs` | `10` | |
| `--lr` | `1e-4` | AdamW. |
| `--val-split` | `0.1` | Fraction of training data held out for validation. |
| `--output-dir` | `saved/models` | Checkpoint directory. |
| `--resume` | `None` | Resume training from a `.pt` checkpoint. |
| `--n-gpu` | `1` | `0` = CPU. |

`ChatGPT` blocks `train` (API-only); use `eval` without `--checkpoint` instead.

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
4. Register the new baseline in `main.py` `REGISTRY`:

```python
'<Name>': (
    'baselines.<Name>.dataset', '<Name>Dataset',
    lambda a: {'<dataset_kwarg>': a.<arg>},   # extra kwargs beyond data_path
    'baselines.<Name>.model',   '<Name>Model',
    lambda a: {'<model_kwarg>': a.<arg>},     # None values are filtered automatically
),
```

5. Implement `model.forward(batch) -> loss` and `model.predict(batch) -> list[list[str]]` (ranked entity MIDs per sample).

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

`main.py` wraps each baseline dataset in `BaseDataLoader` directly (so method-specific fields are preserved):

```python
from base import BaseDataLoader
from data_loader.data_loaders import webqsp_collate

loader = BaseDataLoader(dataset, batch_size=32, shuffle=True,
                        validation_split=0.1, num_workers=1,
                        collate_fn=webqsp_collate)
val_loader = loader.split_validation()  # returns a separate DataLoader
```

`webqsp_collate` returns `dict[str, list]` — every field, including method-specific ones, stays as a Python list (no tensors). `WebQSPDataLoader` in `data_loaders.py` is a convenience wrapper for the base dataset only; use `BaseDataLoader` directly when working with baseline-specific datasets.

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
