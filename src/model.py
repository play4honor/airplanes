import torch
from lightning import pytorch as pl
import torchtune

from data import Tokenizer


class FlightDiffusionModel(pl.LightningModule):

    def __init__(
        self,
        tokenizer: Tokenizer,
        d_model: int,
        depth: int,
        n_heads: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # For inference
        self.tokenizer = tokenizer
        self.mask_idx = tokenizer.mapping["<MASK>"]

        self.n_embeddings = len(tokenizer)
        self.d_model = d_model
        self.ff_dim = d_model * 2  # Or whatever
        self.depth = depth
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        self.embedding = torch.nn.Embedding(self.n_embeddings, self.d_model)
        self.position_encoding = torchtune.modules.RotaryPositionalEmbeddings(
            dim=d_model // n_heads,
            max_seq_len=self.max_seq_len,
        )

        self.transformer = torchtune.modules.TransformerDecoder(
            tok_embeddings=self.embedding,
            layers=torch.nn.ModuleList(
                [
                    torchtune.modules.TransformerSelfAttentionLayer(
                        attn=torchtune.modules.MultiHeadAttention(
                            embed_dim=self.d_model,
                            num_heads=self.n_heads,
                            num_kv_heads=self.n_heads,
                            head_dim=self.d_model // self.n_heads,
                            q_proj=torch.nn.Linear(self.d_model, self.d_model),
                            k_proj=torch.nn.Linear(self.d_model, self.d_model),
                            v_proj=torch.nn.Linear(self.d_model, self.d_model),
                            output_proj=torch.nn.Linear(self.d_model, self.d_model),
                            pos_embeddings=(self.position_encoding if i == 0 else None),
                            is_causal=False,
                        ),
                        mlp=torchtune.modules.FeedForward(
                            gate_proj=torch.nn.Linear(self.d_model, self.ff_dim),
                            down_proj=torch.nn.Linear(self.ff_dim, self.d_model),
                            up_proj=torch.nn.Linear(self.d_model, self.ff_dim),
                        ),
                        sa_norm=torchtune.modules.RMSNorm(self.d_model),
                        mlp_norm=torchtune.modules.RMSNorm(self.d_model),
                    )
                    for i in range(self.depth)
                ]
            ),
            max_seq_len=self.max_seq_len,
            num_heads=self.n_heads,
            head_dim=self.d_model / self.n_heads,
            norm=torchtune.modules.RMSNorm(self.d_model),
            output=torch.nn.Linear(self.d_model, self.n_embeddings),
        )

    def _mask_batch(self, x: torch.Tensor, t: float):
        mask_positions = torch.rand_like(x) < t
        x[mask_positions] = self.mask_idx
        return x, mask_positions

    def loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: torch.Tensor,
        t: float,
    ) -> torch.Tensor:
        # inputs: (n, s, v)
        # targets: (n, s)
        # loss_mask: (n, s)
        unmasked_loss = torch.nn.functional.cross_entropy(
            inputs, targets, reduction="none"
        )
        return (unmasked_loss * loss_mask).sum() / t

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.transformer(x)
        return x

    def step(self, stage, x: dict[str, torch.Tensor]) -> torch.Tensor:
        pass


if __name__ == "__main__":

    from data import Tokenizer

    tokenizer = Tokenizer(["a", "b", "c", "d"])

    model = FlightDiffusionModel(
        tokenizer=tokenizer,
        d_model=32,
        depth=4,
        n_heads=4,
        max_seq_len=32,
    )

    x = torch.randint(0, 4, [16, 32])

    y = model(x)
    print(y.shape)
    print(y)
