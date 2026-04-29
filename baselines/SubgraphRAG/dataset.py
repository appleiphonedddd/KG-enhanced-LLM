import json
from data_loader.base_webqsp_dataset import WebQSPDataset


class SubgraphRAGDataset(WebQSPDataset):
    def __init__(self, data_path, subgraph_path=None):
        super().__init__(data_path)
        for sample in self.samples:
            sample['scored_triples'] = []
        if subgraph_path:
            self._merge_triples(subgraph_path)

    def _merge_triples(self, subgraph_path):
        with open(subgraph_path) as f:
            index = {item['id']: item.get('scored_triples', []) for item in json.load(f)}
        for sample in self.samples:
            sample['scored_triples'] = index.get(sample['id'], [])
