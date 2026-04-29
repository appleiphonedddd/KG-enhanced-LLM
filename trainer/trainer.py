import logging
import torch
from base.base_trainer import BaseTrainer
from utils.metric import f1_score, hits_at_1

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    """
    General trainer for trainable KGQA baselines (EmbedKGQA, NSM, RRP, …).

    Batches from WebQSPDataLoader are dict-of-lists.  The model receives the
    whole batch dict and must expose two methods:
      - forward(batch) -> loss tensor          (called during training)
      - predict(batch) -> list[list[str]]      (called during validation;
                                                each inner list is ranked
                                                predicted entity MIDs/names)

    For baselines that need a separate criterion, pass it explicitly;
    otherwise the model's forward() is expected to return the loss directly.
    """

    def __init__(self, model, optimizer, config, device,
                 data_loader, valid_data_loader=None,
                 criterion=None, lr_scheduler=None):
        super().__init__(model, optimizer, config)
        self.device = device
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler


    def _train_epoch(self, epoch) -> dict:
        self.model.train()
        total_loss = 0.0

        for batch in self.data_loader:
            self.optimizer.zero_grad()
            loss = self._compute_loss(batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        log = {'loss': total_loss / len(self.data_loader)}

        if self.valid_data_loader is not None:
            val_log = self._eval_epoch(epoch)
            log.update({f'val_{k}': v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log


    def _compute_loss(self, batch) -> torch.Tensor:
        """
        If a separate criterion was provided, call model(batch) -> output
        and criterion(output, batch).  Otherwise assume model(batch) -> loss.
        """
        output = self.model(batch)
        if self.criterion is not None:
            return self.criterion(output, batch)
        return output


    def _eval_epoch(self, epoch) -> dict:
        self.model.eval()
        all_hits, all_f1 = [], []

        with torch.no_grad():
            for batch in self.valid_data_loader:
                # model.predict returns list[list[str]] — ranked entity MIDs
                predictions = self.model.predict(batch)
                for preds, answers in zip(predictions, batch['answers']):
                    golds = [a['mid'] for a in answers]
                    top1 = preds[0] if preds else ''
                    all_hits.append(hits_at_1(top1, golds))
                    all_f1.append(f1_score(preds, golds))

        n = len(all_hits)
        return {
            'hits@1': sum(all_hits) / n,
            'f1':     sum(all_f1)   / n,
        }
