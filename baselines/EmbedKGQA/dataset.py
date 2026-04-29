import json
from data_loader.base_webqsp_dataset import WebQSPDataset


class EmbedKGQADataset(WebQSPDataset):
    def __init__(self, data_path, candidates_path=None):
        super().__init__(data_path)
        for sample in self.samples:
            sample['entity_candidates'] = []
        if candidates_path:
            self._merge_candidates(candidates_path)

    def _merge_candidates(self, candidates_path):
        with open(candidates_path) as f:
            index = {item['id']: item.get('candidates', []) for item in json.load(f)}
        for sample in self.samples:
            sample['entity_candidates'] = index.get(sample['id'], [])
