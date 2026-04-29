import json
from data_loader.base_webqsp_dataset import WebQSPDataset


class ReliableReasoningPathDataset(WebQSPDataset):
    def __init__(self, data_path, rrp_path=None):
        super().__init__(data_path)
        for sample in self.samples:
            sample['semantic_paths'] = []
            sample['structural_paths'] = []
        if rrp_path:
            self._merge_paths(rrp_path)

    def _merge_paths(self, rrp_path):
        with open(rrp_path) as f:
            index = {item['id']: item for item in json.load(f)}
        for sample in self.samples:
            entry = index.get(sample['id'], {})
            sample['semantic_paths'] = entry.get('semantic_paths', [])
            sample['structural_paths'] = entry.get('structural_paths', [])
