from lightning import pytorch as pl
import polars as pol
from torch.utils.data import DataLoader, random_split
import torch
import yaml

from src.data import FlightsDataset, Tokenizer
from src.model import FlightDiffusionModel

with open("./config.yaml", "r") as f:
    config = yaml.safe_load(f)

pl.seed_everything(seed=config["compute_params"]["seed"])

# Set up data -------

data = pol.read_parquet("./data/prepared_data.parquet")

model = FlightDiffusionModel.load_from_checkpoint(
    "./runs/lightning_logs/version_3/checkpoints/epoch=2-step=5055.ckpt",
    weights_only=False,
)
model.eval()

ds = FlightsDataset(
    data,
    max_length=config["data_params"]["max_length"],
    tokenizer=model.tokenizer,
)

_, _, test_ds = random_split(ds, config["data_params"]["splits"])

test_dl = DataLoader(
    test_ds,
    batch_size=1,
    num_workers=config["compute_params"]["num_dl_workers"],
)


if __name__ == "__main__":

    # batch = next(iter(test_dl))
    # batch = {
    #     k: v.to(model.device) if isinstance(v, torch.Tensor) else v
    #     for k, v in batch.items()
    # }

    # Leaving from Des Moines
    masked_inputs = torch.full(
        [1, 90], fill_value=model.mask_idx, dtype=torch.int64, device=model.device
    )
    masked_inputs[0, 0] = model.tokenizer.mapping["<SOS>"]
    masked_inputs[0, 1] = model.tokenizer.mapping["AIRPORT=DSM"]

    # original_inputs = model.tokenizer.from_index(batch["event_sequence"])
    # masked_inputs, masked_positions = model._mask_batch(batch["event_sequence"], t=0.5)

    # [n, s, v]

    SEQUENCE_SIZE = 90
    N_STEPS = 20

    for i in range(20):

        masked_positions = masked_inputs == model.mask_idx
        # [n, s, v]
        out = model(masked_inputs)
        out = out + torch.where(
            masked_positions.unsqueeze(-1).expand(-1, -1, out.shape[-1]),
            torch.zeros_like(out),
            torch.full_like(out, float("-inf")),
        )
        # [n, s]
        max_probs, likeliest_tokens = torch.max(out, dim=-1)
        among_highest = torch.argsort(max_probs, dim=-1, descending=True)[:, :5]
        print(among_highest)
        print(masked_inputs.shape)
        print(likeliest_tokens.shape)
        print(among_highest.max())
        likeliest_tokens[among_highest]
        # masked_inputs[among_highest] = likeliest_tokens[among_highest]
        break
