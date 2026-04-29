import json
from data_loader.base_webqsp_dataset import WebQSPDataset


class ToGDataset(WebQSPDataset):
    def __init__(self, data_path, tog_path=None):
        super().__init__(data_path)
        for sample in self.samples:
            sample['beam_paths'] = []
        if tog_path:
            self._merge_paths(tog_path)

    def _merge_paths(self, tog_path):
        with open(tog_path) as f:
            index = {item['id']: item.get('paths', []) for item in json.load(f)}
        for sample in self.samples:
            sample['beam_paths'] = index.get(sample['id'], [])
