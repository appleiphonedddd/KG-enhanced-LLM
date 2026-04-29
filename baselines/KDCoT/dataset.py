import json
from data_loader.base_webqsp_dataset import WebQSPDataset


class KDCoTDataset(WebQSPDataset):
    def __init__(self, data_path, kdcot_path=None):
        super().__init__(data_path)
        for sample in self.samples:
            sample['triples'] = []
        if kdcot_path:
            self._merge_triples(kdcot_path)

    def _merge_triples(self, kdcot_path):
        with open(kdcot_path) as f:
            index = {item['id']: item.get('knowledge', []) for item in json.load(f)}
        for sample in self.samples:
            sample['triples'] = index.get(sample['id'], [])
