import logging
from abc import abstractmethod
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class BaseTrainer:
    """
    Abstract base for all trainable KGQA baselines.

    Subclasses must implement _train_epoch(epoch) -> dict,
    returning a log dict that contains at least 'loss'.
    """

    def __init__(self, model, optimizer, config):
        self.model = model
        self.optimizer = optimizer
        self.config = config

        cfg = config.get('trainer', {})
        self.epochs = cfg.get('epochs', 10)
        self.save_period = cfg.get('save_period', 1)
        self.save_dir = Path(cfg.get('save_dir', 'saved/models'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.start_epoch = 1

        resume = cfg.get('resume')
        if resume:
            self._resume_checkpoint(resume)

    @abstractmethod
    def _train_epoch(self, epoch) -> dict:
        ...

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            log = self._train_epoch(epoch)
            log_str = '  '.join(f'{k}: {v:.4f}' for k, v in log.items())
            logger.info(f'Epoch {epoch}/{self.epochs}  {log_str}')
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch):
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }
        path = self.save_dir / f'checkpoint-epoch{epoch}.pt'
        torch.save(state, path)
        logger.info(f'Checkpoint saved: {path}')

    def _resume_checkpoint(self, path):
        state = torch.load(path, weights_only=True)
        self.start_epoch = state['epoch'] + 1
        self.model.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        logger.info(f'Resumed from checkpoint: {path}  (epoch {state["epoch"]})')
