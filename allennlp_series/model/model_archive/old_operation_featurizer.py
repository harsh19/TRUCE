import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from allennlp_series.model.model_archive.operations_configs import *


class OperationFeaturizer(nn.Module):
    def __init__(
        self, inp_length=12, operations_set_type: str = "all", use_signum: bool = False
    ):
        super().__init__()
        assert operations_set_type in ["all", "configa", "configb"]
        operations_set_type_to_operations_dict = {
            "all": operations_all,
            "configa": operations_configa,
            "configb": operations_configb,
        }
        self.operations = torch.stack(
            operations_set_type_to_operations_dict[operations_set_type]
        )
        # num-chanenls, weight size of each
        self.operations = self.operations.unsqueeze(
            1
        )  # num-chanenls, 1, weight size of each
        self.conv_weights = nn.Parameter(0.3 * torch.ones(self.operations.size()[0]))
        self.num_features = inp_length - 2
        self.out_channels = self.conv_weights.size()[0]
        self.use_signum = use_signum
        self.num_operators = self.operations.size()[0]
        print(
            "[OperationFeaturizer] list(model.parameters()) : ", list(self.parameters())
        )
        print("[OperationFeaturizer]: conv_weights=", self.conv_weights)

    def _normalize(self, features: torch.Tensor):
        # return ( features-features.min() ) / ( features.max()-features.min()+1 )
        # return  features / ( np.abs(features).max()+1 )
        maxval, idx = torch.max(torch.abs(features), dim=1)
        return features / (maxval.view(-1, 1) + 1)

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
        out = out * self.conv_weights.unsqueeze(0).unsqueeze(2)
        return out

    def forward(self, series: torch.Tensor):
        norm_series = self._normalize(series)  # copy. 1,length
        out = self._apply_operation(norm_series)
        if np.random.rand() < 0.05:
            print(self.operations)
            print(self.conv_weights)
        return out

    def get_l1_loss(self):
        return torch.sum(torch.abs(self.conv_weights))


"""
Two operators composed together
Also takes in the num_classes, and outputs class scores on computed features -- l1 loss is computed on these weights
"""


class OperationFeaturizerTwoLayer(nn.Module):
    def __init__(
        self,
        inp_length=12,
        operations_set_type: str = "all",
        use_signum: bool = False,
        num_classes: int = 5,
        use_class_wise_l1: bool = False,
    ):
        super().__init__()
        assert operations_set_type in ["all", "configa", "configb"]
        operations_set_type_to_operations_dict = {
            "all": operations_all,
            "configa": operations_configa,
            "configb": operations_configb,
        }
        self.operations = torch.stack(
            operations_set_type_to_operations_dict[operations_set_type]
        )
        # num-chanenls, weight size of each
        self.operations = self.operations.unsqueeze(
            1
        )  # num-chanenls, 1, weight size of each
        self.num_features = inp_length - 4
        self.use_signum = use_signum
        self.num_classes = num_classes
        self.conv_weights = nn.Parameter(
            0.3
            * torch.ones(
                (num_classes, self.operations.size()[0], self.operations.size()[0])
            )
        )
        self.num_operators = self.out_channels = self.operations.size()[0]
        self.use_class_wise_l1 = use_class_wise_l1
        print(
            "[OperationFeaturizer] list(model.parameters()) : ", list(self.parameters())
        )
        print("[OperationFeaturizer]: conv_weights=", self.conv_weights)

    def _normalize(self, features: torch.Tensor):
        # return ( features-features.min() ) / ( features.max()-features.min()+1 )
        # return  features / ( np.abs(features).max()+1 )
        maxval, idx = torch.max(torch.abs(features), dim=1)
        # print("maxval: ", maxval.size())
        # print("features :", features.size() )
        return features / (maxval.view(-1, 1) + 1)

    def _apply_operation(self, values: torch.Tensor):  # values: batchsize, length
        # F.conv1d(input, self.weight, self.bias, self.stride)
        bs = values.size()[0]
        out = F.conv1d(values.unsqueeze(1), self.operations)
        # out: bs, out-channels, out-length
        ## out-length = in-length - 2
        ## out-channels = num of operations
        out = out.detach()  # keeping conv filters fixed
        if self.use_signum:
            out[out > 0] = 1
            out[out < 0] = -1
        # print("bs = ", bs)
        # print("out: ", out.size())
        # print("num_operators = ", self.num_operators)
        out_re = out.view(bs * self.num_operators, 1, -1)  # bs*num, 1, inp-length-1
        # print("_apply_operation: out_re:", out_re.size())
        out2 = F.conv1d(out_re, self.operations)  # out2: bs*num, num, inp-length-2
        # print("_apply_operation: out2:", out2.size())
        out2 = out2.view(
            bs, self.num_operators, self.num_operators, -1
        )  # bs, num, num, inp-length-2
        # print("_apply_operation: out2:", out2.size())
        # print("_apply_operation: conv_weights:", self.conv_weights.size())
        # self.conv_weights: classes, num, num
        # bs, classes, num, num, feat-size
        out2 = self.conv_weights.unsqueeze(0).unsqueeze(4) * out2.unsqueeze(1)
        # print("_apply_operation: out2 final:", out2.size())
        return out2

    def forward(self, series: torch.Tensor):
        norm_series = self._normalize(series)  # copy. 1,length
        out = self._apply_operation(norm_series)
        if np.random.rand() < 0.05:
            operations = self.operations.data.cpu().numpy()
            conv_weights = self.conv_weights.data.cpu().numpy()
            i = 0
            for (
                conv_weights_i
            ) in (
                conv_weights
            ):  # conv_weights_i per class. conv_weights_i: numops x numops
                all_scores = []
                # print(self.operations.size(), conv_weights_i.shape)
                j = 0
                for conv_weights_ij in conv_weights_i:
                    k = 0
                    for conv_weights_ijk in conv_weights_ij:
                        # print(self.operations[j], self.operations[k])
                        # print(conv_weights_ijk)
                        all_scores.append(
                            [conv_weights_ijk, operations[j], operations[k]]
                        )
                        k += 1
                    j += 1
                print(" i = ", i)
                print(sorted(all_scores, key=lambda x: -x[0]))
                i += 1
        return out

    def get_l1_loss(self):
        if self.use_class_wise_l1:
            # do it class-wise ?
            return torch.sum(
                torch.sum(
                    torch.abs(self.conv_weights).view(self.num_classes, -1), dim=1
                )
            )
        else:
            return torch.sum(torch.abs(self.conv_weights))


"""
Take as input during init specific operator choice in contrast to above which runs for all
"""


class OperationFeaturizerTwoLayerChoice(nn.Module):
    def __init__(
        self,
        inp_length=12,
        operations_set_type: str = "all",
        use_signum: bool = False,
        num_classes: int = 5,
        choice_num: int = None,
        choice_num2: int = None,
    ):
        super().__init__()
        assert operations_set_type in ["all", "configa", "configb"]
        operations_set_type_to_operations_dict = {
            "all": operations_all,
            "configa": operations_configa,
            "configb": operations_configb,
        }
        self.operations = torch.stack(
            operations_set_type_to_operations_dict[operations_set_type]
        )
        self.operations2 = torch.stack(
            operations_set_type_to_operations_dict[operations_set_type]
        )
        if choice_num is not None:
            self.operations = self.operations[choice_num : choice_num + 1]
            self.operations2 = self.operations2[choice_num2 : choice_num2 + 1]
            print(" ===> self.operations = ", self.operations)
            print(" ===> self.operations2 = ", self.operations2)
        # num-chanenls, weight size of each
        self.operations = self.operations.unsqueeze(
            1
        )  # num-chanenls, 1, weight size of each
        self.operations2 = self.operations2.unsqueeze(
            1
        )  # num-chanenls, 1, weight size of each
        self.num_features = inp_length - 4
        self.use_signum = use_signum
        self.num_classes = num_classes
        self.conv_weights = nn.Parameter(
            0.3
            * torch.ones(
                (num_classes, self.operations.size()[0], self.operations.size()[0])
            )
        )
        self.num_operators = self.out_channels = self.operations.size()[0]
        # self.use_class_wise_l1 = use_class_wise_l1
        print(
            "[OperationFeaturizer] list(model.parameters()) : ", list(self.parameters())
        )
        print("[OperationFeaturizer]: conv_weights=", self.conv_weights)

    def _normalize(self, features: torch.Tensor):
        # return ( features-features.min() ) / ( features.max()-features.min()+1 )
        # return  features / ( np.abs(features).max()+1 )
        maxval, idx = torch.max(torch.abs(features), dim=1)
        return features / (maxval.view(-1, 1) + 1)

    def _apply_operation(self, values: torch.Tensor):  # values: batchsize, length
        # F.conv1d(input, self.weight, self.bias, self.stride)
        bs = values.size()[0]
        out = F.conv1d(values.unsqueeze(1), self.operations)
        # out: bs, out-channels, out-length
        ## out-length = in-length - 2
        ## out-channels = num of operations
        out = out.detach()  # keeping conv filters fixed
        if self.use_signum:
            out[out > 0] = 1
            out[out < 0] = -1
        # print("bs = ", bs)
        # print("out: ", out.size())
        # print("num_operators = ", self.num_operators)
        out_re = out.view(bs * self.num_operators, 1, -1)  # bs*num, 1, inp-length-1
        # print("_apply_operation: out_re:", out_re.size())
        out2 = F.conv1d(out_re, self.operations2)  # out2: bs*num, num, inp-length-2
        # print("_apply_operation: out2:", out2.size())
        out2 = out2.view(
            bs, self.num_operators, self.num_operators, -1
        )  # bs, num, num, inp-length-2
        # print("_apply_operation: out2:", out2.size())
        # print("_apply_operation: conv_weights:", self.conv_weights.size())
        # self.conv_weights: classes, num, num
        # bs, classes, num, num, feat-size
        out2 = self.conv_weights.unsqueeze(0).unsqueeze(4) * out2.unsqueeze(1)
        # print("_apply_operation: out2 final:", out2.size())
        # num=1 since only 1 operation
        return out2

    def forward(self, series: torch.Tensor):
        norm_series = self._normalize(series)  # copy. 1,length
        out = self._apply_operation(norm_series)
        if np.random.rand() < 0.05:
            operations = self.operations.data.cpu().numpy()
            conv_weights = self.conv_weights.data.cpu().numpy()
            i = 0
            for (
                conv_weights_i
            ) in (
                conv_weights
            ):  # conv_weights_i per class. conv_weights_i: numops x numops
                all_scores = []
                # print(self.operations.size(), conv_weights_i.shape)
                j = 0
                for conv_weights_ij in conv_weights_i:
                    k = 0
                    for conv_weights_ijk in conv_weights_ij:
                        # print(self.operations[j], self.operations[k])
                        # print(conv_weights_ijk)
                        all_scores.append(
                            [conv_weights_ijk, operations[j], operations[k]]
                        )
                        k += 1
                    j += 1
                print(" i = ", i)
                print(sorted(all_scores, key=lambda x: -x[0]))
                i += 1
        return out

    def get_l1_loss(self):
        # return torch.zeros(1)
        assert False
        # if self.use_class_wise_l1:
        #     # do it class-wise ?
        #     return torch.sum( torch.sum(torch.abs(self.conv_weights).view(self.num_classes, -1), dim=1) )
        # else:
        #     return torch.sum(torch.abs(self.conv_weights))


"""
This class is never run in training. Only if we want to analyze outputs.
"""


class OperationFeaturizerTwoLayerOutputsOnly(nn.Module):
    def __init__(
        self,
        inp_length=12,
        operations_set_type: str = "all",
        use_signum: bool = False,
        num_classes: int = 5,
        use_class_wise_l1: bool = False,
    ):
        super().__init__()
        assert operations_set_type in ["all", "configa", "configb"]
        operations_set_type_to_operations_dict = {
            "all": operations_all,
            "configa": operations_configa,
            "configb": operations_configb,
        }
        self.operations = torch.stack(
            operations_set_type_to_operations_dict[operations_set_type]
        )
        # num-chanenls, weight size of each
        self.operations = self.operations.unsqueeze(
            1
        )  # num-chanenls, 1, weight size of each
        self.num_features = inp_length - 4
        self.use_signum = use_signum
        self.num_classes = num_classes
        self.conv_weights = nn.Parameter(
            0.3
            * torch.ones(
                (num_classes, self.operations.size()[0], self.operations.size()[0])
            )
        )  ## <-- not used
        self.num_operators = self.out_channels = self.operations.size()[0]
        self.use_class_wise_l1 = use_class_wise_l1
        print(
            "[OperationFeaturizer] list(model.parameters()) : ", list(self.parameters())
        )
        print("[OperationFeaturizer]: conv_weights=", self.conv_weights)

    def _normalize(self, features: torch.Tensor):
        # return ( features-features.min() ) / ( features.max()-features.min()+1 )
        # return  features / ( np.abs(features).max()+1 )
        maxval, idx = torch.max(torch.abs(features), dim=1)
        return features / (maxval.view(-1, 1) + 1)

    def _apply_operation(self, values: torch.Tensor):  # values: batchsize, length
        # F.conv1d(input, self.weight, self.bias, self.stride)
        bs = values.size()[0]
        out = F.conv1d(values.unsqueeze(1), self.operations)
        # out: bs, out-channels, out-length
        ## out-length = in-length - 2
        ## out-channels = num of operations
        out = out.detach()  # keeping conv filters fixed
        if self.use_signum:
            out[out > 0] = 1
            out[out < 0] = -1
        # print("bs = ", bs)
        # print("out: ", out.size())
        # print("num_operators = ", self.num_operators)
        out_re = out.view(bs * self.num_operators, 1, -1)  # bs*num, 1, inp-length-1
        # print("_apply_operation: out_re:", out_re.size())
        out2 = F.conv1d(out_re, self.operations)  # out2: bs*num, num, inp-length-2
        # print("_apply_operation: out2:", out2.size())
        out2 = out2.view(
            bs, self.num_operators, self.num_operators, -1
        )  # bs, num, num, inp-length-2
        # print("_apply_operation: out2:", out2.size())
        # print("_apply_operation: conv_weights:", self.conv_weights.size())
        # self.conv_weights: classes, num, num
        # bs, classes, num, num, feat-size
        # --- # out2 = self.conv_weights.unsqueeze(0).unsqueeze(4) * out2.unsqueeze(1) ###3 --> commenting out this part
        # print("_apply_operation: out2 final:", out2.size())
        return out2

    def forward(self, series: torch.Tensor):
        norm_series = self._normalize(series)  # copy. 1,length
        out = self._apply_operation(norm_series)
        if np.random.rand() < 0.05:
            operations = self.operations.data.cpu().numpy()
            conv_weights = self.conv_weights.data.cpu().numpy()
            i = 0
            for (
                conv_weights_i
            ) in (
                conv_weights
            ):  # conv_weights_i per class. conv_weights_i: numops x numops
                all_scores = []
                # print(self.operations.size(), conv_weights_i.shape)
                j = 0
                for conv_weights_ij in conv_weights_i:
                    k = 0
                    for conv_weights_ijk in conv_weights_ij:
                        # print(self.operations[j], self.operations[k])
                        # print(conv_weights_ijk)
                        all_scores.append(
                            [conv_weights_ijk, operations[j], operations[k]]
                        )
                        k += 1
                    j += 1
                print(" i = ", i)
                print(sorted(all_scores, key=lambda x: -x[0]))
                i += 1
        return out

    # def get_l1_loss(self):
    #     if self.use_class_wise_l1:
    #         # do it class-wise ?
    #         return torch.sum( torch.sum(torch.abs(self.conv_weights).view(self.num_classes, -1), dim=1) )
    #     else:
    #         return torch.sum(torch.abs(self.conv_weights))


#####################
