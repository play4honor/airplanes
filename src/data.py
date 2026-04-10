import torch
from torch.utils.data import Dataset
import polars as pl
import numpy as np


class Tokenizer:

    def __init__(self, tokens: list[str]):

        self.mapping = {token: i for i, token in enumerate(sorted(set(tokens)))}
        features = set(token.split("=")[0] for token in self.mapping.keys())
        for feature in features:
            self.mapping[f"{feature}=UNK"] = len(self.mapping)
        self.mapping["<MASK>"] = len(self.mapping)

        self.inverse_mapping = {v: k for k, v in self.mapping.items()}

    def to_index(self, token_sequence: list[str]):

        return torch.tensor(
            [
                self.mapping.get(token, self.mapping[f"{token.split('=')[0]}=UNK"])
                for token in token_sequence
            ],
            dtype=torch.long,
        )

    def from_index(self, idx: torch.Tensor):

        if not hasattr(self, "inverse_mapping"):
            self.inverse_mapping = {v: k for k, v in self.mapping.items()}

        return np.vectorize(self.inverse_mapping.get)(idx.numpy(force=True))

    def __len__(self):
        return len(self.mapping)


class FlightsDataset(Dataset):

    def __init__(self, data: pl.DataFrame, max_length: int, tokenizer: Tokenizer):
        super().__init__()

        self.data = data
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return self.data.height

    def __getitem__(self, idx):
        row = self.data.row(idx, named=True)
        indices = self.tokenizer.to_index(row["flight_info"])[: self.max_length]
        indices = torch.nn.functional.pad(
            indices,
            [0, self.max_length - indices.shape[0]],
            value=self.tokenizer.mapping["<EOS>"],
        )
        return {
            # "flight_date": [row["FL_DATE"]],
            "tail_number": [row["tail_number"]],
            "event_sequence": indices,
        }


if __name__ == "__main__":

    import polars as pl

    data = pl.read_parquet("./data/prepared_data.parquet")
    print(data.with_columns(_len=pl.col("flight_info").list.len()).max()["_len"])

    vocabulary = (
        data.select(pl.col("flight_info").explode().alias("tokens"))
        .unique()["tokens"]
        .to_list()
    )

    tokenizer = Tokenizer(vocabulary)
    ds = FlightsDataset(
        data,
        max_length=86,
        tokenizer=tokenizer,
    )

    print(ds[500])
    print(tokenizer.mapping)
