from base import BaseDataLoader
from data_loader.base_webqsp_dataset import WebQSPDataset


def webqsp_collate(batch):
    return {k: [s[k] for s in batch] for k in batch[0]}


class WebQSPDataLoader(BaseDataLoader):
    def __init__(self, data_path, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        dataset = WebQSPDataset(data_path)
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=webqsp_collate)
