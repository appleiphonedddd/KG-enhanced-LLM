import json
from data_loader.base_webqsp_dataset import WebQSPDataset


class KVMemDataset(WebQSPDataset):
    def __init__(self, data_path, memories_path=None):
        super().__init__(data_path)
        for sample in self.samples:
            sample['memories'] = []
        if memories_path:
            self._merge_memories(memories_path)

    def _merge_memories(self, memories_path):
        with open(memories_path) as f:
            index = {item['id']: item.get('memories', []) for item in json.load(f)}
        for sample in self.samples:
            sample['memories'] = index.get(sample['id'], [])
