from data_loader.base_webqsp_dataset import WebQSPDataset


class ChatGPTDataset(WebQSPDataset):
    def __init__(self, data_path, use_cot=False):
        super().__init__(data_path)
        for sample in self.samples:
            sample['prompt'] = self._make_prompt(sample['question'], use_cot)

    def _make_prompt(self, question, use_cot):
        if use_cot:
            return f"{question}\nLet's think step by step."
        return question
