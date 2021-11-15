import torch.nn as nn
import torch.nn.functional as F

from allennlp_series.model.model_archive.operations_configs import *

"""
Mostly a reimplementation of OperationFeaturizer above but allows easier modular use 
"""


def normalize_with_maxval(values: torch.Tensor):
    if len(values.size()) == 2:
        maxval, idx = torch.max(torch.abs(values), dim=1)
    else:
        bs = values.size()[0]
        maxval, idx = torch.max(torch.abs(values.view(bs, -1)), dim=-1)
    # print("maxval: ", maxval.size() )
    # print("values: ", values.size())
    if len(values.size()) == 2:
        return values / (maxval.view(-1, 1) + 1)
    else:
        ret = values / (maxval.view(-1, 1, 1) + 1)
        # print("ret: ", ret.size())
        return ret


class Operator(nn.Module):
    def __init__(
        self, operator_name: str = "default", inp_length=12, use_l1_loss: bool = True
    ):
        super().__init__()
        self.use_l1_loss = use_l1_loss
        self.num_features = None  # number of output features
        self.inp_length = inp_length  # number of input features
        self.operator_name = operator_name

    def _apply_operation(self, values: torch.Tensor):  # values: batchsize, length
        raise NotImplementedError

    def forward(self, values: torch.Tensor):
        raise NotImplementedError
        # return torch.tensor(0.)

    def get_l1_loss(self):
        raise NotImplementedError


class ConvOperators(Operator):
    def __init__(
        self,
        inp_length=12,
        operations_type: str = "all",
        use_signum: bool = False,
        operator_name: str = "conv_operators",
        use_l1_loss: bool = True,
    ):
        super().__init__(operator_name=operator_name, inp_length=inp_length)
        assert operations_type in ["all", "configa", "configb"]
        self.operations_type = operations_type
        operations_set_type_to_operations_dict = {
            "all": operations_all,
            "configa": operations_configa,
            "configb": operations_configb,
        }
        self.operations = torch.stack(
            operations_set_type_to_operations_dict[operations_type]
        )
        # num-chanenls, weight size of each
        self.operations = self.operations.unsqueeze(
            1
        )  # num-chanenls, 1, weight size of each
        self.num_features = inp_length - 2
        self.out_channels = self.operations.size()[0]  # self.conv_weights.size()[0]
        self.num_operators = self.operations.size()[0]
        self.use_signum = use_signum
        self.use_l1_loss = use_l1_loss
        print(
            "[OperationFeaturizer] ConvOperators: list(model.parameters()) : ",
            list(self.parameters()),
        )

    def _apply_operation(self, values: torch.Tensor):  # values: batchsize, length
        # F.conv1d(input, self.weight, self.bias, self.stride)
        out = F.conv1d(values.unsqueeze(1), self.operations)
        # out: bs, out-channels, out-length
        ## out-length = in-length - 2
        ## out-channels = num of operations
        out = out.detach()  # keeping conv filters fixed
        if self.use_signum:
            out[out > 0] = 1
            out[out < 0] = -1
        # print(out.size(), self.conv_weights.size())
        # out = out * self.conv_weights.unsqueeze(0).unsqueeze(2)
        return out

    def forward(self, values: torch.Tensor):
        """
        :param values:
        :return:
        """
        out = self._apply_operation(values)
        # if np.random.rand()<0.01:
        #    print(self.operations)
        #    # print(self.conv_weights)
        return out

    def get_l1_loss(self):
        return torch.sum(torch.abs(self.conv_weights))


class ConvOperatorsChoice(Operator):
    def __init__(
        self,
        inp_length=12,
        operations_type: str = "all",
        use_signum: bool = False,
        operator_name: str = "conv_operators",
        use_l1_loss: bool = True,
    ):
        super().__init__(operator_name=operator_name, inp_length=inp_length)
        assert operations_type in ["all", "configa", "configb"]
        self.operations_type = operations_type
        operations_set_type_to_operations_dict = {
            "all": operations_all,
            "configa": operations_configa,
            "configb": operations_configb,
        }
        self.operations = torch.stack(
            operations_set_type_to_operations_dict[operations_type]
        )
        # num-chanenls, weight size of each
        self.operations = self.operations.unsqueeze(
            1
        )  # num-chanenls, 1, weight size of each
        self.num_features = inp_length - 2
        self.out_channels = (
            1  ####****** self.operations.size()[0] #self.conv_weights.size()[0]
        )
        self.num_operators = self.operations.size()[0]
        self.use_signum = use_signum
        self.use_l1_loss = use_l1_loss
        print(
            "[OperationFeaturizer] ConvOperatorsChoice: list(model.parameters()) : ",
            list(self.parameters()),
        )

    def _apply_operation(
        self, values: torch.Tensor, choice_num: int = None
    ):  # values: batchsize, length
        # print("ConvOperatorsChoice ====> choice_num = ", choice_num)
        # print("ConvOperatorsChoice ====> values:", values.size())
        if choice_num is not None:
            operations = self.operations[choice_num : choice_num + 1]
        else:
            operations = self.operations
        out = F.conv1d(values.unsqueeze(1), operations)
        out = out.detach()  # keeping conv filters fixed
        if self.use_signum:
            out[out > 0] = 1
            out[out < 0] = -1
        return out

    def forward(self, values: torch.Tensor, choice_num: int = None):
        """
        :param values:
        :return:
        """
        out = self._apply_operation(values, choice_num=choice_num)
        return out

    def get_l1_loss(self):
        return torch.sum(torch.abs(self.conv_weights))


class AccumulatorOperator(Operator):
    def __init__(
        self,
        inp_length=12,
        operations_type: str = "all",
        use_signum: bool = False,
        operator_name: str = "conv_operators",
        use_l1_loss: bool = True,
    ):
        super().__init__(operator_name=operator_name, inp_length=inp_length)
        assert operations_type in ["max", "mean", "min", "max_min_mean"]
        self.num_operators = len(operations_type.split("_"))
        self.operations_type = operations_type
        self.num_features = 1
        if operations_type == "max_min_mean":
            self.num_features = 3
        self.use_signum = use_signum
        self.use_l1_loss = use_l1_loss
        print(
            "[OperationFeaturizer] [AccumulatorOperator] list(model.parameters()) : ",
            list(self.parameters()),
        )

    def _apply_operation(self, values: torch.Tensor):  # values: batchsize, length
        if self.operations_type == "max":
            out, _ = torch.max(values, dim=-1)  # ,....,length -> ,....,
            out = out.unsqueeze(-1)
        elif self.operations_type == "min":
            out, _ = torch.min(values, dim=-1)  # ,....,length -> ,....,
            out = out.unsqueeze(-1)
        elif self.operations_type == "mean":
            out = torch.mean(values, dim=-1)  # ,....,length -> ,....,
            out = out.unsqueeze(-1)
        elif self.operations_type == "max_min_mean":
            out1, _ = torch.max(values, dim=-1)
            out2, _ = torch.min(values, dim=-1)
            out3 = torch.mean(values, dim=-1)
            out = torch.cat(
                [out1.unsqueeze(-1), out2.unsqueeze(-1), out3.unsqueeze(-1)], dim=-1
            )
        else:
            out = values
        return out

    def forward(self, values: torch.Tensor):
        """
        :param values:
        :return:
        """
        out = self._apply_operation(values)
        return out


class AccumulatorOperatorChoice(Operator):
    def __init__(
        self,
        inp_length=12,
        operations_type: str = "max_min_mean",
        use_signum: bool = False,
        operator_name: str = "accum_choice_operators",
    ):
        super().__init__(operator_name=operator_name, inp_length=inp_length)
        assert operations_type in ["max_min_mean"]
        self.operations_type = operations_type
        self.num_features = 1
        self.num_operators = 3
        self.use_signum = use_signum
        assert not use_signum
        print(
            "[OperationFeaturizer] [AccumulatorOperatorChoice] list(model.parameters()) : ",
            list(self.parameters()),
        )

    def _apply_operation(
        self, values: torch.Tensor, choice_num: int = None
    ):  # values: batchsize, length
        # print("[AccumulatorOperatorChoice]: values=",values)
        if choice_num == 0:
            out, _ = torch.max(values, dim=-1)  # ,....,length -> ,....,
            out = out.unsqueeze(-1)
        elif choice_num == 1:
            out, _ = torch.min(values, dim=-1)  # ,....,length -> ,....,
            out = out.unsqueeze(-1)
        elif choice_num == 2:
            out = torch.mean(values, dim=-1)  # ,....,length -> ,....,
            out = out.unsqueeze(-1)
        else:
            out = None
        # print("[AccumulatorOperatorChoice]: choice_num=",choice_num," out=",out)
        return out

    def forward(self, values: torch.Tensor, choice_num: int = None):
        """
        :param values:
        :return:
        """
        out = self._apply_operation(values, choice_num)
        return out


######
# creat this extra featurizer in clf code
# this returns, batch, new-series, length=12-or-whatever
# add this to each batch. batch,num_series=2,lenth -> batch,num_series+#new-series,length
# run the normal 3 layer or two layer program. more than normal number of features are expected.
# final clf weights need to be adjusted


class VertTwoSeriesConvOperator(Operator):
    def __init__(
        self,
        inp_length=2,
        operations_type: str = "all",
        use_signum: bool = False,
        operator_name: str = "vert_conv_operators",
    ):
        super().__init__(operator_name=operator_name, inp_length=inp_length)
        assert inp_length == 2  ## since two series vert operator
        assert operations_type in ["all", "vertconfiga", "vertconfigb"]
        self.operations_type = operations_type
        operations_set_type_to_operations_dict = {
            "all": None,
            "vertconfiga": vert_two_operations_configa,
            "vertconfigb": None,
        }
        self.operations = torch.stack(
            operations_set_type_to_operations_dict[operations_type]
        )
        # num-chanenls, weight size of each
        self.operations = self.operations.unsqueeze(
            1
        )  # num-chanenls, 1, weight size of each
        self.num_features = inp_length - 1
        self.out_channels = self.operations.size()[0]  # self.conv_weights.size()[0]
        self.num_operators = self.operations.size()[0]
        self.use_signum = use_signum
        print(
            "[OperationFeaturizer] [VertTwoSeriesConvOperator] list(model.parameters()) : ",
            list(self.parameters()),
        )

    def _apply_operation(self, values: torch.Tensor, choice_num: int = None):
        # values: batchsize , num_series_in_each, length
        # F.conv1d(input, self.weight, self.bias, self.stride)
        # print("[VertTwoSeriesConvOperator]: [_apply_operation]: values:", values.size())
        batch, num_series_in_each, length = (
            values.size()[0],
            values.size()[1],
            values.size()[2],
        )
        values = values.permute(
            0, 2, 1
        ).contiguous()  # batch, length, num_series_in_each
        values = values.view(
            -1, num_series_in_each
        )  # bs=batch*length, num_series_in_each
        out = F.conv1d(values.unsqueeze(1), self.operations)
        # out: bs, out-channels, out-length=1
        ## out-length = num_series_in_each - 1
        ## out-channels = num of operations
        out = out.detach()  # keeping conv filters fixed
        if self.use_signum:
            out[out > 0] = 1
            out[out < 0] = -1
        # print(out.size(), self.conv_weights.size())
        # out = out * self.conv_weights.unsqueeze(0).unsqueeze(2)
        out = out.view(batch, length, -1)  # batch,length,out-channels*1
        out = out.permute(0, 2, 1)  # batch,out-channels*1,length
        # for each batch element, there are out-channels more time-series now
        return out

    def forward(self, values: torch.Tensor, choice_num: int = None):
        out = self._apply_operation(values)
        return out

    def get_l1_loss(self):
        return torch.sum(torch.abs(self.conv_weights))


class VertTwoSeriesConvOperatorChoice(Operator):
    def __init__(
        self,
        inp_length=2,
        operations_type: str = "all",
        use_signum: bool = False,
        operator_name: str = "vert_conv_operators",
    ):
        super().__init__(operator_name=operator_name, inp_length=inp_length)
        assert inp_length == 2  ## since two series vert operator
        assert operations_type in ["all", "vertconfiga", "vertconfigb"]
        self.operations_type = operations_type
        operations_set_type_to_operations_dict = {
            "all": None,
            "vertconfiga": vert_two_operations_configa,
            "vertconfigb": None,
        }
        self.operations = torch.stack(
            operations_set_type_to_operations_dict[operations_type]
        )
        # num-chanenls, weight size of each
        self.operations = self.operations.unsqueeze(
            1
        )  # num-chanenls, 1, weight size of each
        self.num_features = inp_length - 1
        self.out_channels = 1  # since choice operator #self.operations.size()[0]
        self.num_operators = self.operations.size()[0]
        self.use_signum = use_signum
        print(
            "[OperationFeaturizer] [VertTwoSeriesConvOperatorChoice] list(model.parameters()) : ",
            list(self.parameters()),
        )

    def _apply_operation(self, values: torch.Tensor, choice_num: int = None):
        # values: batchsize , num_series_in_each, length
        # print("values : ", values.size())
        # print("choice_num : ", choice_num)
        # F.conv1d(input, self.weight, self.bias, self.stride)
        # print("[VertTwoSeriesConvOperator]: [_apply_operation]: values:", values.size())
        batch, num_series_in_each, length = (
            values.size()[0],
            values.size()[1],
            values.size()[2],
        )
        values = values.permute(
            0, 2, 1
        ).contiguous()  # batch, length, num_series_in_each
        values = values.view(
            -1, num_series_in_each
        )  # bs=batch*length, num_series_in_each
        operations = self.operations[choice_num : choice_num + 1]
        out = F.conv1d(values.unsqueeze(1), operations)
        # out: bs, out-channels, out-length=1
        ## out-length = num_series_in_each - 1
        ## out-channels = num of operations
        out = out.detach()  # keeping conv filters fixed
        if self.use_signum:
            out[out > 0] = 1
            out[out < 0] = -1
        out = out.view(batch, length, -1)  # batch,length,out-channels*1
        out = out.permute(0, 2, 1)  # batch,out-channels*1,length
        return out

    def forward(self, values: torch.Tensor, choice_num: int = None):
        out = self._apply_operation(values, choice_num=choice_num)
        return out
