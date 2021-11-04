# Seq2SeqEncoder

from typing import Dict, List, Tuple, Union, Any

import torch
import numpy as np

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn.util import get_text_field_mask
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import Perplexity
from allennlp_series.common.constants import *
from torch.nn.modules.rnn import LSTMCell
from torch.nn.modules.linear import Linear
import torch.nn.functional as F
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from allennlp_series.training.metrics import CocovalsMeasures
from allennlp_series.training.metrics.diversity_evals import DiversityEvals
import os
import torch.nn as nn

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import math
from torch.nn import (
    Linear,
    ReLU,
    CrossEntropyLoss,
    Sequential,
    Conv1d,
    MaxPool1d,
    MaxPool2d,
    Module,
    Softmax,
    BatchNorm1d,
    Dropout,
)


def get_fft(x):
    rfft_x = np.fft.rfft(x)
    ret = np.hstack([rfft_x.real, rfft_x.imag])
    ret = torch.tensor(ret)
    if torch.cuda.is_available():
        ret = ret.cuda()
    return ret


class CNNNet(Module):
    def __init__(self, inp_len=12, output_dim=None):
        super(CNNNet, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv1d(1, 4, kernel_size=3, stride=1, padding=0),
            # ) #,
            BatchNorm1d(4),
            ReLU(inplace=True),
            MaxPool1d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv1d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm1d(4),
            ReLU(inplace=True),
            MaxPool1d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(Linear(4 * 2, output_dim))

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        # print("after cnn layers : ",x.size())
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


@Model.register("seq2seq_encoder")
class Seq2SeqEncoder(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        embedding_dim: int = 5,
        model_type="1layer",
        dropout: float = None,
        initializer: InitializerApplicator = None,
        **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)

        self._input_dim = input_dim = embedding_dim
        output_dim = embedding_dim
        self.model_type = model_type
        assert model_type in [
            "1layer",
            "2layer",
            "lstm",
            "fft",
            "cnn",
            "lstm_extended",
            "lstm_norm",
            "transformer",
        ]
        if model_type in ["1layer", "2layer"]:
            if model_type == "1layer":
                self.layer1 = nn.Linear(input_dim, output_dim)
            else:
                h = int(math.sqrt(input_dim))
                self.layer1 = nn.Sequential(
                    nn.Linear(input_dim, h), nn.ReLU(), nn.Linear(h, output_dim)
                )
            print("[Seq2SeqEncoder] self.layer1  = ", self.layer1)
        elif model_type in ["lstm", "lstm_extended", "lstm_norm"]:
            if model_type == "lstm_extended":
                self._decoder_cell = torch.nn.LSTM(
                    embedding_dim, output_dim, batch_first=True
                )
            else:
                self._decoder_cell = torch.nn.LSTM(1, output_dim, batch_first=True)
                # or perhaps represent in form of 1s, 10s, 100s
            print("[Seq2SeqEncoder] self._decoder_cell  = ", self._decoder_cell)
        elif model_type in ["cnn"]:
            self.net = CNNNet(input_dim, output_dim)
            print("[Seq2SeqEncoder][CNN] self.net  = ", self.net)
        elif model_type in ["fft"]:
            self.layer1 = nn.Linear(2 * (1 + input_dim // 2), output_dim)
            print("[Seq2SeqEncoder] self.fft: layer1  = ", self.layer1)
        elif model_type in ["transformer"]:
            self.net = nn.Tra(input_dim, output_dim)
            print("[Seq2SeqEncoder][CNN] self.net  = ", self.net)

        if initializer is not None:
            initializer(self)

    def forward(self, series) -> torch.Tensor:  # type: ignore

        if self.model_type in ["1layer", "2layer"]:
            ret = self.layer1(series)
        elif self.model_type in ["fft"]:
            vals = get_fft(series)
            # print("--->>> fft: vals: ", series.size())
            # print("--->>> fft: vals: ", vals.size())
            # print("--->>> fft: self.layer1: ", self.layer1)
            ret = self.layer1(vals)
        elif self.model_type in ["cnn"]:
            # print("series : ", series.size())
            vals = self.net(series.unsqueeze(1))
            # print("--->>> cnn: vals: ", vals.size())
            # print("--->>> fft: vals: ", vals.size())
            # print("--->>> fft: self.layer1: ", self.layer1)
            ret = vals  # self.layer1(vals)
        else:
            if self.model_type in ["lstm_extended"]:
                # series: bs,len
                # series.unsqueeze(2) : bs,len,1
                # print("[Seq2SeqEncoder]: series: ", series.size() )
                lstm_out, (ht, ct) = self._decoder_cell(
                    series.unsqueeze(2).repeat(1, 1, self._input_dim)
                )
            else:
                # series: bs,len
                # series.unsqueeze(2) : bs,len,1
                # print("[Seq2SeqEncoder]: series: ", series.size() )
                if self.model_type == "lstm_norm":
                    series = (series - 50.0) / 50.0
                lstm_out, (ht, ct) = self._decoder_cell(series.unsqueeze(2))

            decoder_hidden_last = ht[-1]
            ret = decoder_hidden_last  # bs, output_size

        # print("[Seq2SeqEncoder]: series = ", series )
        # print("[Seq2SeqEncoder]: ret: ", ret.size() )

        return ret


# repeat:
# >>> y
# tensor([[[1],
#          [2],
#          [3]],
#
#         [[7],
#          [8],
#          [7]]])
# >>> y.shape
# torch.Size([2, 3, 1])
# >>> z=y.repeat(1,1,3)
# >>> z
# tensor([[[1, 1, 1],
#          [2, 2, 2],
#          [3, 3, 3]],
#
#         [[7, 7, 7],
#          [8, 8, 8],
#          [7, 7, 7]]])
# >>> z.shape
# torch.Size([2, 3, 3])
# >>> z[0][0]
# tensor([1, 1, 1])
