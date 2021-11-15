from allennlp_series.model.model_archive.operations import *

import torch.nn as nn
import torch
from allennlp_series.common.constants import *


## Sample modular class
class ThreeLayerProgram(nn.Module):
    """
    create operator layer 1
    create operator layer 2
    create accumulator layer
    """

    #
    """
    fwd pass
    - inp: bs, length
    - normalize: bs, length
    - layer1: bs, channels, length
    - reshape: bs*channels, length
    - layer2: bs*channels, channels, length
    - reshape: bs, channels, channels, length
    - accumulator: bs, channels, channels, acc_feats
    - reshape: bs, channels*channels*acc_feats 
    """

    def __init__(
        self,
        inp_length=12,
        operations_conv_type: str = "all",
        operations_acc_type: str = "max",
        use_signum: bool = False,
        use_l1_loss: bool = True,
        run_without_norm: bool = False,
        series_type: str = SERIES_TYPE_SINGLE,
    ):
        super().__init__()
        assert series_type in SERIES_TYPES_LIST
        self.series_type = series_type
        self.run_without_norm = run_without_norm
        if run_without_norm:
            raise NotImplementedError
        self.layer1 = ConvOperators(
            inp_length=inp_length,
            operations_type=operations_conv_type,
            use_signum=use_signum,
        )
        self.layer1_numfeats = self.layer1.num_features
        self.num_operators = self.layer1.out_channels
        self.layer2 = ConvOperators(
            inp_length=self.layer1_numfeats,
            operations_type=operations_conv_type,
            use_signum=use_signum,
        )
        self.layer2_numfeats = self.layer2.num_features
        self.layer3 = AccumulatorOperator(
            inp_length=self.layer2_numfeats, operations_type=operations_acc_type
        )
        self.use_l1_loss = use_l1_loss
        self.num_features = (
            self.num_operators * self.num_operators * self.layer3.num_features
        )

    def forward(self, series: torch.Tensor):
        bs = series.size()[0]
        if self.series_type == SERIES_TYPE_MULTI:
            norm_series = normalize_with_maxval(series)
            series_row_cnt = series.size()[1]
            bs_old = bs
            bs = bs_old * series_row_cnt
            norm_series = norm_series.view(bs, -1)
        else:
            norm_series = normalize_with_maxval(series)  # copy. 1,length
        # print(" series = ", series)
        # print(" norm_series = ", norm_series)
        # norm_series = normalize_with_maxval( series ) # copy. 1,length
        out = self.layer1(norm_series)
        out_re = out.view(bs * self.num_operators, -1)  # bs*num, inp-length-1
        out2 = self.layer2(out_re)
        out2 = out2.view(
            bs, self.num_operators, self.num_operators, -1
        )  # bs, num, num, featsize
        out3 = self.layer3(out2)
        out3 = out3.view(bs, -1)
        if self.series_type == SERIES_TYPE_MULTI:
            # out3 = out3.view(bs_old,series_row_cnt,-1)
            out3 = out3.view(bs_old, -1)
            # print("out3: ", out3.size())
        return out3


## Sample modular class
class TwoLayerProgram(nn.Module):
    """
    create operator layer 1
    create operator layer 2
    create accumulator layer
    """

    #
    """
    fwd pass
    - inp: bs, length
    - normalize: bs, length
    - layer1: bs, channels, length
    - reshape: bs*channels, length
    - layer2: bs*channels, channels, length
    - reshape: bs, channels, channels, length
    - accumulator: bs, channels, channels, acc_feats
    - reshape: bs, channels*channels*acc_feats 
    """

    def __init__(
        self,
        inp_length=12,
        operations_conv_type: str = "all",
        operations_acc_type: str = "max",
        use_signum: bool = False,
        use_l1_loss: bool = True,
        run_without_norm: bool = False,
        series_type: str = SERIES_TYPE_SINGLE,
    ):
        super().__init__()
        assert series_type in SERIES_TYPES_LIST
        self.series_type = series_type
        self.run_without_norm = run_without_norm
        if run_without_norm:
            raise NotImplementedError
        self.layer1 = ConvOperators(
            inp_length=inp_length,
            operations_type=operations_conv_type,
            use_signum=use_signum,
        )
        self.layer1_numfeats = self.layer1.num_features
        self.num_operators = self.layer1.out_channels
        self.layer3 = AccumulatorOperator(
            inp_length=self.layer1_numfeats, operations_type=operations_acc_type
        )
        self.use_l1_loss = use_l1_loss
        self.num_features = self.num_operators * self.layer3.num_features
        print()
        print("============ TwoLayerProgram ==========")
        print()

    def forward(self, series: torch.Tensor):
        bs = series.size()[0]
        if self.series_type == SERIES_TYPE_MULTI:
            norm_series = normalize_with_maxval(series)
            series_row_cnt = series.size()[1]
            bs_old = bs
            bs = bs_old * series_row_cnt
            norm_series = norm_series.view(bs, -1)
        else:
            norm_series = normalize_with_maxval(series)  # copy. 1,length
        # print(" series = ", series)
        # print(" norm_series = ", norm_series)
        # print("norm_series: ", norm_series.size())
        out = self.layer1(norm_series)
        out_re = out.view(bs * self.num_operators, -1)  # bs*num, inp-length-1
        # out2 = self.layer1(out_re)
        # out2 = out2.view(bs, self.num_operators, self.num_operators, -1)  # bs, num, num, featsize
        out2 = out.view(bs, self.num_operators, -1)  # bs, num, featsize
        out3 = self.layer3(out2)
        out3 = out3.view(bs, -1)
        # print("out3 = ", out3)
        if self.series_type == SERIES_TYPE_MULTI:
            # out3 = out3.view(bs_old,series_row_cnt,-1)
            out3 = out3.view(bs_old, -1)
            # print("out3: ", out3.size())
            # print("out3 reshaped = ", out3)
        return out3


if __name__ == "__main__":

    model = ThreeLayerProgram(operations_conv_type="configb")
    arr = torch.tensor(
        [
            [
                0.0100,
                0.0333,
                -0.0266,
                -0.0166,
                0.0333,
                0.0133,
                -0.0300,
                0.0033,
                0.0100,
                0.0333,
                -0.0266,
                -0.0166,
            ]
        ]
    )
    print(arr.size())
    feats = model(arr)
    print(feats.size())

    # model = TwoLayerProgram(operations_conv_type='configb')
    # arr = torch.tensor([[0.0100, 0.0333, -0.0266, -0.0166, 0.0333, 0.0133, -0.0300,
    #                      0.0033, 0.0100, 0.0333, -0.0266, -0.0166]])
    # print(arr.size())
    # feats = model(arr)
    # print(feats.size())
