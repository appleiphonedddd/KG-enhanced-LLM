import json
from data_loader.base_webqsp_dataset import WebQSPDataset


class NSMDataset(WebQSPDataset):
    def __init__(self, data_path, nsm_path=None):
        super().__init__(data_path)
        for sample in self.samples:
            sample['hop_entities'] = []
        if nsm_path:
            self._merge_hops(nsm_path)

    def _merge_hops(self, nsm_path):
        with open(nsm_path) as f:
            index = {item['id']: item.get('entities', []) for item in json.load(f)}
        for sample in self.samples:
            sample['hop_entities'] = index.get(sample['id'], [])
