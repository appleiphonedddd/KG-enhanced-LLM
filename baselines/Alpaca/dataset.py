from data_loader.base_webqsp_dataset import WebQSPDataset

_ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{question}\n\n### Response:"
)


class AlpacaDataset(WebQSPDataset):
    def __init__(self, data_path):
        super().__init__(data_path)
        for sample in self.samples:
            sample['prompt'] = _ALPACA_TEMPLATE.format(question=sample['question'])
