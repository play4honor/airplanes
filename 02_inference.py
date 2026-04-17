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

SEQUENCE_SIZE = 170
STEP_SIZE = 2

# Set up data -------

data = pol.read_parquet("./data/prepared_data.parquet")

model = FlightDiffusionModel.load_from_checkpoint(
    "./runs/lightning_logs/version_6/checkpoints/epoch=2-step=15165.ckpt",
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
        [1, SEQUENCE_SIZE],
        fill_value=model.mask_idx,
        dtype=torch.int64,
        device=model.device,
    )
    masked_inputs[0, 0] = model.tokenizer.mapping["<SOS>"]
    masked_inputs[0, 1] = model.tokenizer.mapping["AIRPORT=LGA"]

    # original_inputs = model.tokenizer.from_index(batch["event_sequence"])
    # masked_inputs, masked_positions = model._mask_batch(batch["event_sequence"], t=0.5)

    # [n, s, v]

    n_masked = (masked_inputs == model.mask_idx).sum(dim=-1)[0].item()

    while n_masked > 0:

        actual_step_size = min(STEP_SIZE, n_masked)

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
        _, highest_positions = torch.sort(max_probs, dim=-1, descending=True)
        sorted_likeliest_tokens = torch.gather(
            likeliest_tokens,
            dim=-1,
            index=highest_positions,
        )
        masked_inputs.scatter_(
            dim=-1,
            index=highest_positions[:, :actual_step_size],
            src=sorted_likeliest_tokens[:, :actual_step_size],
        )
        n_masked -= actual_step_size

        print(model.tokenizer.from_index(masked_inputs))
