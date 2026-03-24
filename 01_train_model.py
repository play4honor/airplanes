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

unique_tokens = (
    data.select(pol.col("flight_info").explode().alias("tokens"))
    .unique()["tokens"]
    .to_list()
)
tokenizer = Tokenizer(unique_tokens)

ds = FlightsDataset(
    data,
    max_length=config["data_params"]["max_length"],
    tokenizer=tokenizer,
)

train_ds, valid_ds, _ = random_split(ds, config["data_params"]["splits"])

train_dl = DataLoader(
    train_ds,
    batch_size=config["train_params"]["batch_size"],
    shuffle=True,
    num_workers=config["compute_params"]["num_dl_workers"],
)
valid_dl = DataLoader(
    valid_ds,
    batch_size=config["train_params"]["batch_size"],
    num_workers=config["compute_params"]["num_dl_workers"],
)

# Set up trainer and model ------

trainer = pl.Trainer(
    accelerator="gpu",
    max_epochs=config["train_params"]["max_epochs"],
    log_every_n_steps=config["compute_params"]["log_every_n_steps"],
    accumulate_grad_batches=config["train_params"]["accumulate_grad_batches"],
    precision=config["compute_params"]["precision"],
    logger=pl.loggers.tensorboard.TensorBoardLogger(save_dir="./runs"),
)
with trainer.init_module():
    model = FlightDiffusionModel(
        tokenizer=tokenizer,
        **config["model_params"],
        lr=config["train_params"]["learning_rate"],
        max_seq_len=config["data_params"]["max_length"],
    )
    model = torch.compile(
        model,
        fullgraph=False,
        options={
            "shape_padding": True,
        },
    )

trainer.fit(
    model,
    train_dataloaders=train_dl,
    val_dataloaders=valid_dl,
)
