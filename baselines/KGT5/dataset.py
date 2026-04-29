from data_loader.base_webqsp_dataset import WebQSPDataset


class KGT5Dataset(WebQSPDataset):
    def __init__(self, data_path):
        super().__init__(data_path)
        for sample in self.samples:
            names = [a['name'] for a in sample['answers'] if a['name']]
            sample['target_text'] = names[0] if names else ''
