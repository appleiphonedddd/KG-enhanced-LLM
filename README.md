# 🚀 KG-enhanced-LLM: Where Large Language Models Meet Knowledge Graphs!

> *"LLMs are smart but prone to hallucinations; KGs are rigorous but lack flexibility. What if we brought them together?"*
> 
> Welcome to **KG-enhanced-LLM**! This isn't just a codebase; it's a carnival for "Knowledge Graph-enhanced Language Models"! 🎉

## 🌟 The "Why" (Project Background)

To all the researchers and developers hustling in the KG-LLM space, have you ever experienced these pain points?
- You want to compare your new model against recent papers, but the old baselines just won't run? 🤯
- Every paper has a different environment and dataset format, and "reproducing" them eats up a whole month? 😭
- You want a unified framework for fair A/B testing, but don't know where to start?

**This is exactly why this project was born!** Our ultimate goal is: **To build the most comprehensive, out-of-the-box baseline reproduction hub for the Knowledge Graph + LLM domain!**

## ✨ Core Features

- 🛠️ **One-Stop Reproduction**: From the classic KG-RAG to the cutting-edge GNN-LLM fusion architectures, we are building them all!
- 📊 **A Fair Arena**: Unified data preprocessing and standardized evaluation metrics, ensuring every model competes on the same starting line.
- 🐍 **Pythonic & Readable**: Say goodbye to academic spaghetti code. We provide a clean, modular architecture.
- 🤝 **Open Source Spirit**: Fully open-source under the Apache License Version 2.0. Everyone is welcome to build together!

## 🥊 The Contender Lineup (Implemented Baselines)

Ladies and gentlemen, please welcome your contestants to the stage! 🎙️

Each baseline is fully implemented under `baselines/<Name>/` and evaluated on **WebQSP** under identical data splits and metrics. Let's meet the cast:

---

### 🏛️ Act 1 — The Embedding Pioneers
*Before transformers ruled the earth, these models showed KG structure could be baked directly into the embedding space.*

| # | Model | Authors | Venue | One-liner |
|---|-------|---------|-------|-----------|
| 1 | **KVMem** | Miller, Fisch, Dodge et al. | EMNLP 2016 | The O.G. memory network — look up facts at read time via key-value attention. Simple, interpretable, still competitive. |
| 2 | **EmbedKGQA** | Saxena, Tripathi & Talukdar | ACL 2020 | Why not just embed the whole KG? Pre-trained RotatE embeddings + a question encoder handle multi-hop QA with pure dot products. |
| 3 | **NSM** | He, Lan, Jiang et al. | SIGIR 2021 | KGQA as a state machine — the model "walks" the graph by updating a distribution over entities at each hop. Interpretable by design. |
| 4 | **TransferNet** | Shi, Cao, Hou et al. | EMNLP 2021 | Every relation gets its own transfer matrix. Multi-hop reasoning is a chain of relation-conditioned transforms over entity scores — fast, transparent, elegant. |
| 5 | **KGT5** | Saxena, Kochsiek & Gemulla | NAACL 2022 | The bridge to the generative era: serialize the local KG subgraph as text, hand it to T5, and let it generate the answer. |

---

### 📡 Act 2 — The Retrieval Veterans
*Retrieve a question-relevant subgraph first, reason over it second. A two-stage pipeline that aged surprisingly well.*

| # | Model | Authors | Venue | One-liner |
|---|-------|---------|-------|-----------|
| 6 | **GraftNet** | Sun, Dhingra, Zaheer et al. | EMNLP 2018 | One of the first to fuse KG triples + Wikipedia passages in a single heterogeneous graph. Structured + unstructured beats either alone. |
| 7 | **PullNet** | Sun, Bedrax-Weiss & Cohen | EMNLP 2019 | GraftNet's smarter sibling: *iteratively* pulls new evidence from the KG and text at each reasoning step. Dynamic retrieval for dynamic reasoning. |
| 8 | **SR** | Zhang, Zhang, Yu et al. | ACL 2022 | Deceptively simple: retrieve the right subgraph first, then do cheap in-subgraph reasoning. Achieved SOTA at ACL 2022 with a lean architecture. |

---

### 🤖 Act 3 — The LLM-Only Challengers
*No KGs, pure language model power. These baselines answer the uncomfortable question: "How much does the KG actually help?"*

| # | Model | Authors | Venue | One-liner |
|---|-------|---------|-------|-----------|
| 9 | **FlanT5** | Chung, Hou, Longpre et al. (Google Brain) | JMLR 2024 | The instruction-tuning revolution in one model. Fine-tuned on 1,800+ tasks with CoT data — generalizes to KGQA zero-shot. |
| 10 | **Alpaca** | Taori, Gulrajani, Zhang et al. (Stanford CRFM) | 2023 | Stanford's scrappy open-source answer to ChatGPT. LLaMA-7B × 52K self-instruct examples. Punches above its weight class. |
| 11 | **LLaMA 2** | Touvron, Martin, Stone et al. (Meta AI) | arXiv 2023 | Meta goes open-weight. 7B / 13B / 70B variants with RLHF alignment. Our strongest open-source LLM-only baseline. |
| 12 | **ChatGPT** | OpenAI | 2022 | The model that changed everything. GPT-3.5 Turbo sets the commercial API ceiling — and helps us isolate how much credit the KG truly deserves. |

---

### 🧠 Act 4 — The LLM × KG Fusion Headliners
*The main event. Parametric knowledge of LLMs meets the structural precision of KGs. This is where the real battle is fought.*

| # | Model | Authors | Venue | One-liner |
|---|-------|---------|-------|-----------|
| 13 | **KDCoT** | — | 2023 | Knowledge-Driven Chain-of-Thought: each intermediate reasoning step is anchored to a verifiable KG fact, catching and correcting hallucinations before they cascade. |
| 14 | **UniKGQA** | Jiang, Zhou, Zhao & Wen (RUC) | ICLR 2023 | One model to rule them all — a single LM jointly handles *which triples are relevant* and *what's the answer* in a unified retrieval-reasoning pipeline. |
| 15 | **ToG** | Sun, Xu, Tang et al. | ICLR 2024 | Think-on-Graph: the LLM becomes an interactive graph explorer, selecting which relation to traverse at each step. Zero extra training — pure LLM reasoning over a live KG. |
| 16 | **KGCoT** | — | 2023 | Knowledge Graph Chain-of-Thought: KG triples are injected at each reasoning step, grounding the chain-of-thought in structured evidence rather than parametric memory. |
| 17 | **RoG** | Luo, Li, Haffari & Pan (Monash + CSIRO) | ICLR 2024 | Reasoning on Graphs: *plan* a relation path first, *retrieve* grounding facts along that path, then *reason* with the LLM. Faithful, interpretable, ICLR 2024 quality. |
| 18 | **SubgraphRAG** | Li, Miao & Pan | 2024 | "Simple is Effective" — the title says it all. A lightweight triple scorer + compact retrieved subgraph fed to an LLM can match far more complex pipelines. |
| 19 | **RRP** | Xiao, Zhou, Zhang, Li, Li & Huang | IEEE TKDE 2026 | Reliable Reasoning Path: distills KG-grounded guidance into LLM reasoning via two complementary modules — *semantic paths* (LLM-driven) and *structural paths* (relation-embedding-driven) — for reasoning that is both fluent and structurally faithful. |

---

*(💡 Got a model you want to see added? Drop an Issue and let's make it happen!)*

## 🛠️ Deployment

Create a virtual environment and install the Python libraries:

```sh
conda env create -f env.yaml
conda activate KG
```

## 🎮 Running Baselines (`main.py`)

All baselines share a single entry point with two subcommands: **`train`** and **`eval`**.

```
python main.py {train,eval} --baseline <NAME> [options]
```

### Quick-start examples

**Train GraftNet** (retrieval-based, needs a prebuilt subgraph file):
```sh
python main.py train --baseline GraftNet \
    --train-data datasets/WebQSP/WebQSP.train.json \
    --subgraph data/graftnet_subgraphs.json \
    --vocab-size 30000 --epochs 20 --batch-size 32 --lr 1e-3
```

**Train NSM** (embedding-based, optional preprocessed hop-entities):
```sh
python main.py train --baseline NSM \
    --train-data datasets/WebQSP/WebQSP.train.json \
    --preprocessed data/nsm_hop_entities.json \
    --num-hops 2 --epochs 30
```

**Train FlanT5** (LLM-only, no extra data needed):
```sh
python main.py train --baseline FlanT5 \
    --model-name google/flan-t5-xl \
    --epochs 5 --batch-size 8 --lr 5e-5
```

**Train RRP** (proposed method, requires an LLM path):
```sh
python main.py train --baseline RRP \
    --train-data datasets/WebQSP/WebQSP.train.json \
    --model-name meta-llama/Llama-2-7b-chat-hf \
    --preprocessed data/rrp_paths.json \
    --num-hops 2 --epochs 10
```

**Evaluate any trained baseline** from a checkpoint:
```sh
python main.py eval --baseline GraftNet \
    --test-data datasets/WebQSP/WebQSP.test.json \
    --checkpoint saved/models/checkpoint-epoch20.pt \
    --subgraph data/graftnet_subgraphs.json \
    --vocab-size 30000
```

**Evaluate ChatGPT** (zero-shot, no checkpoint needed):
```sh
python main.py eval --baseline ChatGPT \
    --test-data datasets/WebQSP/WebQSP.test.json \
    --api-key $OPENAI_API_KEY
```

### Common options

| Flag | Default | Description |
|------|---------|-------------|
| `--baseline` | — | **Required.** One of the 19 baseline names above. |
| `--train-data` | `datasets/WebQSP/WebQSP.train.json` | Training split path. |
| `--test-data` | `datasets/WebQSP/WebQSP.test.json` | Test split path. |
| `--preprocessed` | `None` | Pre-extracted JSON for memories / triples / paths (KVMem, NSM, KDCoT, RRP, …). |
| `--subgraph` | `None` | Pre-extracted subgraph JSON (GraftNet, PullNet, SR, UniKGQA, SubgraphRAG). |
| `--model-name` | model default | HuggingFace model name/path for LLM-based baselines. |
| `--epochs` | `10` | Number of training epochs. |
| `--batch-size` | `32` | Batch size. |
| `--lr` | `1e-4` | Learning rate (AdamW). |
| `--val-split` | `0.1` | Fraction of training data held out for validation. |
| `--n-gpu` | `1` | Number of GPUs (`0` = CPU). |
| `--output-dir` | `saved/models` | Checkpoint directory. |
| `--resume` | `None` | Path to a `.pt` checkpoint to resume training from. |
| `--checkpoint` | `None` | Path to a `.pt` checkpoint to load for eval. |

Run `python main.py train --help` or `python main.py eval --help` for the full option list.