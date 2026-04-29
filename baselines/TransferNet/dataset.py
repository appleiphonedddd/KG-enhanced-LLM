import json
from data_loader.base_webqsp_dataset import WebQSPDataset


class TransferNetDataset(WebQSPDataset):
    def __init__(self, data_path, transfernet_path=None):
        super().__init__(data_path)
        for sample in self.samples:
            sample['entity_candidates'] = []
        if transfernet_path:
            self._merge_candidates(transfernet_path)

    def _merge_candidates(self, transfernet_path):
        with open(transfernet_path) as f:
            index = {item['id']: item.get('candidates', []) for item in json.load(f)}
        for sample in self.samples:
            sample['entity_candidates'] = index.get(sample['id'], [])
