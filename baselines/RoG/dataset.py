import json
from data_loader.base_webqsp_dataset import WebQSPDataset


class RoGDataset(WebQSPDataset):
    def __init__(self, data_path, rog_path=None):
        super().__init__(data_path)
        for sample in self.samples:
            sample['reasoning_paths'] = []
        if rog_path:
            self._merge_paths(rog_path)

    def _merge_paths(self, rog_path):
        with open(rog_path) as f:
            index = {item['id']: item.get('graph', []) for item in json.load(f)}
        for sample in self.samples:
            sample['reasoning_paths'] = index.get(sample['id'], [])
