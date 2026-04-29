from data_loader.base_webqsp_dataset import WebQSPDataset

_SYSTEM = "You are a helpful assistant that answers questions concisely and accurately."
_LLAMA2_TEMPLATE = "<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{question} [/INST]"


class LLaMA2Dataset(WebQSPDataset):
    def __init__(self, data_path):
        super().__init__(data_path)
        for sample in self.samples:
            sample['prompt'] = _LLAMA2_TEMPLATE.format(
                system=_SYSTEM,
                question=sample['question'],
            )
