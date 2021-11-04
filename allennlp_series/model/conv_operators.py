import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

operations_init_type1 = [
    0.2 * torch.tensor([1.0 / 3, 1.0 / 3, 1.0 / 3]),
    0.2 * torch.tensor([-1.0, 1.0, 0.0]),
    0.2 * torch.tensor([1.0, -1.0, 0.0]),
]


class SimpleConvOperator(nn.Module):
    def __init__(
        self,
        inp_length=12,
        operator_name: str = "conv_operators",
        use_l1_loss: bool = False,
        kernel_size=3,
        provided_kernel=None,
        provided_bias=None,
        fixed_module: bool = False,
        pad_len=0,
    ):

        super().__init__()
        self.fixed_module = fixed_module
        if provided_kernel:  # provided_kernel: kernel- dimensional values
            assert False
            # assert len(provided_kernel) == kernel_size
            # provided_kernel = torch.tensor(provided_kernel).unsqueeze(0).unsqueeze(0)
            # if torch.cuda.is_available():
            #     provided_kernel = provided_kernel.cuda()
            # self.conv_weights = nn.Parameter(provided_kernel)
            # if provided_bias:
            #     self.bias = nn.Parameter(torch.tensor(provided_bias))
            # else:
            #     self.bias = nn.Parameter(torch.zeros(1)) - 0.05
            # if torch.cuda.is_available():
            #     self.bias = self.bias.cuda()
        else:
            # print(2*torch.rand(41)-1)
            # self.conv_weights = nn.Parameter(20*torch.rand(1,1,kernel_size)-5)
            self.conv_weights = nn.Parameter(torch.rand(1, 1, kernel_size) - 0.5)
            assert not fixed_module
            self.bias = nn.Parameter(torch.rand(1))
        self.tanh = nn.Tanh()

        self.num_features = inp_length - 2
        self.out_channels = self.conv_weights.size()[0]
        self.num_operators = self.conv_weights.size()[0]
        self.use_l1_loss = use_l1_loss
        self.operator_name = operator_name
        self.pad_len = pad_len

        print()
        print(
            "==PARAMS== [conv_opertions] SimpleConvOperator: list(model.parameters()) : ",
            list(self.parameters()),
        )
        print()


    def forward(self, values: torch.Tensor, deb: bool = False):
        """
        :param values:
        :return:
        """
        conv_weights = self.tanh(self.conv_weights)
        if self.pad_len > 0:
            values = F.pad(values, [0, self.pad_len])
        out = F.conv1d(values, conv_weights, bias=self.bias)
        if deb or np.random.rand() < 0.001:
            print(
                "[SimpleConvOperator] operator_name = ",
                self.operator_name,
                " self.conv_weights: ",
                self.conv_weights.view(-1),
            )
            print(
                "[SimpleConvOperator] operator_name = ",
                self.operator_name,
                " conv_weights: ",
                conv_weights,
            )
            print(
                "[SimpleConvOperator] operator_name = ",
                self.operator_name,
                " out = ",
                out,
            )
            print()
        return out

    def get_l1_loss(self):
        return torch.sum(torch.abs(self.conv_weights))

    def __str__(self):
        return "SimpleConvOperator num_features={} out_channels={} ".format(
            self.num_features, self.out_channels
        )

    def get_useful_partitions(self):
        ret = {}
        name = self.operator_name
        # print("self.conv_weights = ", self.conv_weights.data.cpu().numpy())
        for j, val in enumerate(self.conv_weights.data.cpu().numpy().reshape(-1)):
            ret[name + "_conv_weights" + str(j)] = float(val)
        return ret
