import torch
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.5
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model

    def forward(self, src: torch.Tensor) -> torch.Tensor:  # , src_mask: torch.Tensor
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        print("-src:", src.size())
        src = src.unsqueeze(2).repeat(1, 1, 200)
        print("-src:", src.size())
        src = self.pos_encoder(src)
        print("-src:", src.size())
        output = self.transformer_encoder(src)  # src_mask
        return output


#
# def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
#     """Generates an upper-triangular matrix of -inf, with zeros on diag."""
#     return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


if __name__ == "__main__":
    t = TransformerModel(200, 2, 200, 2)
    a = torch.rand((3, 12))  # bs=3, seqlen=4
    out = t(a.t())
    print(out.size())
    # https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# python transf_models.py
# -src: torch.Size([12, 3])
# -src: torch.Size([12, 3, 200])
# -src: torch.Size([12, 3, 200])
# torch.Size([12, 3, 200])
