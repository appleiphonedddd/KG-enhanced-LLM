import json
from data_loader.base_webqsp_dataset import WebQSPDataset

_EMPTY_SUBGRAPH = {'entities': [], 'tuples': []}


class UniKGQADataset(WebQSPDataset):
    def __init__(self, data_path, subgraph_path=None):
        super().__init__(data_path)
        for sample in self.samples:
            sample['subgraph'] = {'entities': [], 'tuples': []}
        if subgraph_path:
            self._merge_subgraph(subgraph_path)

    def _merge_subgraph(self, subgraph_path):
        with open(subgraph_path) as f:
            index = {item['id']: item.get('subgraph', _EMPTY_SUBGRAPH) for item in json.load(f)}
        for sample in self.samples:
            sample['subgraph'] = index.get(sample['id'], _EMPTY_SUBGRAPH)
