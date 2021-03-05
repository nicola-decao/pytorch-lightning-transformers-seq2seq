import os

from torch.utils.data import Dataset


class Seq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_path,
        data_split,
        max_length=128,
    ):
        super().__init__()
        self.tokenizer = tokenizer

        with open(os.path.join(data_path, f"{data_split}.source")) as fs, open(
            os.path.join(data_path, f"{data_split}.target")
        ) as ft:
            self.data = [(s.strip(), t.strip()) for s, t in zip(fs, ft)]

        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return {
            "src": self.data[item][0],
            "trg": self.data[item][1],
        }

    def collate_fn(self, batch):
        batches = {
            "{}_{}".format(name, k): v
            for name in (
                "src",
                "trg",
            )
            for k, v in self.tokenizer(
                [b[name] for b in batch],
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            ).items()
        }
        batches["trg_input_ids"][:, 0] = self.tokenizer.eos_token_id
        batches["raw"] = batch
        return batches
