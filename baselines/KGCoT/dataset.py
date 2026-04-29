import json
from data_loader.base_webqsp_dataset import WebQSPDataset


class KGCoTDataset(WebQSPDataset):
    def __init__(self, data_path, kgcot_path=None):
        super().__init__(data_path)
        for sample in self.samples:
            sample['reasoning_steps'] = []
        if kgcot_path:
            self._merge_steps(kgcot_path)

    def _merge_steps(self, kgcot_path):
        with open(kgcot_path) as f:
            index = {item['id']: item.get('steps', []) for item in json.load(f)}
        for sample in self.samples:
            sample['reasoning_steps'] = index.get(sample['id'], [])
