from data_loader.base_webqsp_dataset import WebQSPDataset


class FlanT5Dataset(WebQSPDataset):
    def __init__(self, data_path):
        super().__init__(data_path)
        for sample in self.samples:
            sample['prompt'] = f"Answer the question: {sample['question']}"
