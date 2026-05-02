"""
KG-Enhanced LLM benchmark CLI.

Train:
  python main.py train --baseline GraftNet \\
      --train-data datasets/WebQSP/WebQSP.train.json \\
      --batch-size 32 --epochs 20 --lr 1e-3 \\
      --subgraph data/graftnet_subgraphs.json

Eval:
  python main.py eval --baseline FlanT5 \\
      --test-data datasets/WebQSP/WebQSP.test.json \\
      --checkpoint saved/checkpoint-epoch10.pt
"""

import argparse
import importlib
import logging
import sys

import torch
from torch.optim import AdamW

from base import BaseDataLoader
from data_loader.data_loaders import webqsp_collate
from trainer.trainer import Trainer
from utils.metric import f1_score, hits_at_1
from utils.util import prepare_device

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s  %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Baseline registry
# Each entry: (dataset_module, dataset_cls, dataset_kwargs_fn,
#              model_module,   model_cls,   model_kwargs_fn)
#
# dataset_kwargs_fn(args) -> dict of extra kwargs (beyond data_path)
# model_kwargs_fn(args)   -> dict of kwargs for the model constructor
# None values are filtered so constructor defaults are respected.
# ---------------------------------------------------------------------------
REGISTRY = {
    # ── Embedding-based ────────────────────────────────────────────────────
    'KVMem': (
        'baselines.KVMem.dataset',  'KVMemDataset',
        lambda a: {'memories_path': a.preprocessed},
        'baselines.KVMem.model',    'KVMemModel',
        lambda a: {'vocab_size': a.vocab_size, 'embed_dim': a.embed_dim,
                   'num_hops': a.num_hops},
    ),
    'EmbedKGQA': (
        'baselines.EmbedKGQA.dataset',  'EmbedKGQADataset',
        lambda a: {'candidates_path': a.preprocessed},
        'baselines.EmbedKGQA.model',    'EmbedKGQAModel',
        lambda a: {'pretrained_model_name': a.model_name,
                   'entity_dim': a.entity_dim},
    ),
    'NSM': (
        'baselines.NSM.dataset',  'NSMDataset',
        lambda a: {'nsm_path': a.preprocessed},
        'baselines.NSM.model',    'NSMModel',
        lambda a: {'pretrained_model_name': a.model_name, 'num_hops': a.num_hops},
    ),
    'TransferNet': (
        'baselines.TransferNet.dataset',  'TransferNetDataset',
        lambda a: {'transfernet_path': a.preprocessed},
        'baselines.TransferNet.model',    'TransferNetModel',
        lambda a: {'pretrained_model_name': a.model_name,
                   'num_hops': a.num_hops},
    ),
    'KGT5': (
        'baselines.KGT5.dataset',  'KGT5Dataset',
        lambda a: {},
        'baselines.KGT5.model',    'KGT5Model',
        lambda a: {'model_name_or_path': a.model_name or 'google/t5-base'},
    ),
    # ── Retrieval-based ────────────────────────────────────────────────────
    'GraftNet': (
        'baselines.GraftNet.dataset',  'GraftNetDataset',
        lambda a: {'subgraph_path': a.subgraph},
        'baselines.GraftNet.model',    'GraftNetModel',
        lambda a: {'vocab_size': a.vocab_size, 'embed_dim': a.embed_dim,
                   'hidden_dim': a.hidden_dim, 'num_gcn_layers': a.num_gcn_layers},
    ),
    'PullNet': (
        'baselines.PullNet.dataset',  'PullNetDataset',
        lambda a: {'subgraph_path': a.subgraph},
        'baselines.PullNet.model',    'PullNetModel',
        lambda a: {'pretrained_model_name': a.model_name},
    ),
    'SR': (
        'baselines.SR.dataset',  'SRDataset',
        lambda a: {'subgraph_path': a.subgraph},
        'baselines.SR.model',    'SRModel',
        lambda a: {'pretrained_model_name': a.model_name,
                   'rel_vocab_size': a.vocab_size or 6000},
    ),
    # ── LLM-only ───────────────────────────────────────────────────────────
    'FlanT5': (
        'baselines.FlanT5.dataset',  'FlanT5Dataset',
        lambda a: {},
        'baselines.FlanT5.model',    'FlanT5Model',
        lambda a: {'model_name_or_path': a.model_name or 'google/flan-t5-xl'},
    ),
    'Alpaca': (
        'baselines.Alpaca.dataset',  'AlpacaDataset',
        lambda a: {},
        'baselines.Alpaca.model',    'AlpacaModel',
        lambda a: {'model_name_or_path': a.model_name or 'tatsu-lab/alpaca-7b'},
    ),
    'LLaMA2': (
        'baselines.LLaMA2.dataset',  'LLaMA2Dataset',
        lambda a: {},
        'baselines.LLaMA2.model',    'LLaMA2Model',
        lambda a: {'model_name_or_path': a.model_name
                   or 'meta-llama/Llama-2-7b-chat-hf'},
    ),
    'ChatGPT': (
        'baselines.ChatGPT.dataset',  'ChatGPTDataset',
        lambda a: {'use_cot': a.use_cot},
        'baselines.ChatGPT.model',    'ChatGPTModel',
        lambda a: {'model': a.model_name or 'gpt-3.5-turbo',
                   'api_key': a.api_key},
    ),
    # ── LLMs + KGs ─────────────────────────────────────────────────────────
    'KDCoT': (
        'baselines.KDCoT.dataset',  'KDCoTDataset',
        lambda a: {'kdcot_path': a.preprocessed},
        'baselines.KDCoT.model',    'KDCoTModel',
        lambda a: {'model_name_or_path': a.model_name},
    ),
    'UniKGQA': (
        'baselines.UniKGQA.dataset',  'UniKGQADataset',
        lambda a: {'subgraph_path': a.subgraph},
        'baselines.UniKGQA.model',    'UniKGQAModel',
        lambda a: {'pretrained_model_name': a.model_name or 'bert-base-uncased'},
    ),
    'ToG': (
        'baselines.ToG.dataset',  'ToGDataset',
        lambda a: {'tog_path': a.preprocessed},
        'baselines.ToG.model',    'ToGModel',
        lambda a: {'model_name_or_path': a.model_name,
                   'beam_width': a.beam_width},
    ),
    'KGCoT': (
        'baselines.KGCoT.dataset',  'KGCoTDataset',
        lambda a: {'kgcot_path': a.preprocessed},
        'baselines.KGCoT.model',    'KGCoTModel',
        lambda a: {'pretrained_model_name': a.model_name or 'bert-base-uncased',
                   'llm_name_or_path': a.llm_name},
    ),
    'RoG': (
        'baselines.RoG.dataset',  'RoGDataset',
        lambda a: {'rog_path': a.preprocessed},
        'baselines.RoG.model',    'RoGPlanner',
        lambda a: {'model_name_or_path': a.model_name},
    ),
    'SubgraphRAG': (
        'baselines.SubgraphRAG.dataset',  'SubgraphRAGDataset',
        lambda a: {'subgraph_path': a.subgraph},
        'baselines.SubgraphRAG.model',    'SubgraphRAGModel',
        lambda a: {'llm_name_or_path': a.llm_name, 'hidden_dim': a.hidden_dim},
    ),
    # ── Proposed ───────────────────────────────────────────────────────────
    'RRP': (
        'baselines.ReliableReasoningPath.dataset',  'ReliableReasoningPathDataset',
        lambda a: {'rrp_path': a.preprocessed},
        'baselines.ReliableReasoningPath.model',    'RRPModel',
        lambda a: {'llm_name_or_path': a.model_name,
                   'glove_dim': a.embed_dim or 300,
                   'hidden_dim': a.hidden_dim or 256,
                   'num_hops': a.num_hops or 2},
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_class(module_path, class_name):
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _filter_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def build_dataset(args, data_path):
    entry = REGISTRY[args.baseline]
    ds_mod, ds_cls_name, ds_kwargs_fn = entry[0], entry[1], entry[2]
    DatasetCls = _load_class(ds_mod, ds_cls_name)
    extra = _filter_none(ds_kwargs_fn(args))
    return DatasetCls(data_path, **extra)


def build_model(args):
    entry = REGISTRY[args.baseline]
    m_mod, m_cls_name, m_kwargs_fn = entry[3], entry[4], entry[5]
    ModelCls = _load_class(m_mod, m_cls_name)
    kwargs = _filter_none(m_kwargs_fn(args))
    return ModelCls(**kwargs)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_train(args):
    if args.baseline == 'ChatGPT':
        logger.error('ChatGPT is API-based and does not support training.')
        sys.exit(1)

    device, gpu_ids = prepare_device(args.n_gpu)

    train_dataset = build_dataset(args, args.train_data)
    train_loader = BaseDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        validation_split=args.val_split,
        num_workers=args.num_workers,
        collate_fn=webqsp_collate,
    )
    val_loader = train_loader.split_validation() if args.val_split > 0 else None

    model = build_model(args)
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    config = {
        'trainer': {
            'epochs': args.epochs,
            'save_period': args.save_period,
            'save_dir': args.output_dir,
            'resume': args.resume,
        }
    }

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config,
        device=device,
        data_loader=train_loader,
        valid_data_loader=val_loader,
    )
    trainer.train()


def cmd_eval(args):
    device, _ = prepare_device(args.n_gpu)

    test_dataset = build_dataset(args, args.test_data)
    test_loader = BaseDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        validation_split=0.0,
        num_workers=args.num_workers,
        collate_fn=webqsp_collate,
    )

    model = build_model(args)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state['model_state'])
        logger.info(f'Loaded checkpoint: {args.checkpoint}')
    model = model.to(device)
    model.eval()

    all_hits, all_f1 = [], []
    with torch.no_grad():
        for batch in test_loader:
            preds_batch = model.predict(batch)
            for preds, answers in zip(preds_batch, batch['answers']):
                golds = [a['mid'] for a in answers]
                top1 = preds[0] if preds else ''
                all_hits.append(hits_at_1(top1, golds))
                all_f1.append(f1_score(preds, golds))

    n = len(all_hits)
    hits = sum(all_hits) / n
    f1 = sum(all_f1) / n
    print(f'\nResults on {args.test_data}  (n={n})')
    print(f'  Hits@1 : {hits:.4f}')
    print(f'  F1     : {f1:.4f}')


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='KG-Enhanced LLM benchmark',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # ── shared args factory ────────────────────────────────────────────────
    def add_shared(p):
        p.add_argument('--baseline', required=True, choices=list(REGISTRY),
                       help='Baseline to run')
        p.add_argument('--batch-size', type=int, default=32)
        p.add_argument('--n-gpu', type=int, default=1,
                       help='Number of GPUs (0 = CPU)')
        p.add_argument('--num-workers', type=int, default=1)
        # Dataset extras
        p.add_argument('--preprocessed',
                       help='Path to preprocessed JSON (memories, triples, paths, …)')
        p.add_argument('--subgraph',
                       help='Path to subgraph JSON (GraftNet, PullNet, SR, UniKGQA, SubgraphRAG)')
        p.add_argument('--use-cot', action='store_true',
                       help='Enable chain-of-thought prompt (ChatGPT only)')
        # Model extras
        p.add_argument('--model-name',
                       help='HuggingFace model name/path for LLM-based baselines')
        p.add_argument('--llm-name',
                       help='Secondary LLM path (KGCoT, SubgraphRAG)')
        p.add_argument('--api-key', help='OpenAI API key (ChatGPT only)')
        p.add_argument('--vocab-size', type=int,
                       help='Vocabulary size (GraftNet, KVMem, SR)')
        p.add_argument('--embed-dim', type=int,
                       help='Embedding dimension')
        p.add_argument('--hidden-dim', type=int,
                       help='Hidden dimension')
        p.add_argument('--num-hops', type=int,
                       help='Number of hops (KVMem, NSM, TransferNet, RRP)')
        p.add_argument('--num-gcn-layers', type=int, default=2,
                       help='GCN layers (GraftNet, PullNet)')
        p.add_argument('--entity-dim', type=int,
                       help='Entity embedding dimension (EmbedKGQA)')
        p.add_argument('--beam-width', type=int, default=3,
                       help='Beam width (ToG)')

    # ── train ──────────────────────────────────────────────────────────────
    train_p = sub.add_parser('train', help='Train a baseline',
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_shared(train_p)
    train_p.add_argument('--train-data',
                         default='datasets/WebQSP/WebQSP.train.json',
                         help='Path to WebQSP training JSON')
    train_p.add_argument('--val-split', type=float, default=0.1,
                         help='Fraction of training data used for validation')
    train_p.add_argument('--epochs', type=int, default=10)
    train_p.add_argument('--lr', type=float, default=1e-4)
    train_p.add_argument('--output-dir', default='saved/models',
                         help='Directory for checkpoints')
    train_p.add_argument('--save-period', type=int, default=1,
                         help='Save checkpoint every N epochs')
    train_p.add_argument('--resume',
                         help='Path to checkpoint to resume training from')

    # ── eval ───────────────────────────────────────────────────────────────
    eval_p = sub.add_parser('eval', help='Evaluate a baseline',
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_shared(eval_p)
    eval_p.add_argument('--test-data',
                        default='datasets/WebQSP/WebQSP.test.json',
                        help='Path to WebQSP test JSON')
    eval_p.add_argument('--checkpoint',
                        help='Path to .pt checkpoint (omit for zero-shot baselines)')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.baseline not in REGISTRY:
        logger.error(f'Unknown baseline: {args.baseline}')
        sys.exit(1)

    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'eval':
        cmd_eval(args)


if __name__ == '__main__':
    main()
