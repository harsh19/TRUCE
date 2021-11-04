import torch
import numpy as np


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


def get_prefixed_operator(typ: str = "middle", inp_length=12):
    if typ == "middle":
        operator = np.zeros(inp_length, dtype=np.float32)
        left = inp_length // 3
        right = (2 * inp_length) // 3
        operator[left:right] = 1
        return torch.tensor(operator)
    elif typ == "begin":
        operator = np.zeros(inp_length, dtype=np.float32)
        right = inp_length // 3
        left = 0
        operator[left:right] = 1
        operator[right : right + 1] = 0.5
        return torch.tensor(operator)
    elif typ == "end":
        operator = np.zeros(inp_length, dtype=np.float32)
        right = inp_length
        left = (2 * inp_length) // 3
        operator[left:right] = 1
        operator[left - 1 : left] = 0.5
        return torch.tensor(operator)
    else:
        raise NotImplementedError


def get_prefixed_operator_new(typ: str = "middle", inp_length=12):
    print("inp_length = ", inp_length)
    if typ == "begin":
        operator = np.zeros(inp_length, dtype=np.float32) - 1.0
        right = inp_length // 3
        left = 0
        operator[left:right] = 1
        # operator[right] = 10
        print("operator = ", operator)
        return torch.tensor(operator)
    elif typ == "middle":
        operator = np.zeros(inp_length, dtype=np.float32) - 1.0
        left = inp_length // 3
        right = (2 * inp_length) // 3
        operator[left:right] = 1
        # operator[right-1] = 10
        # operator[left-1] = 10
        return torch.tensor(operator)
    elif typ == "end":
        operator = np.zeros(inp_length, dtype=np.float32) - 1.0
        right = inp_length
        left = (2 * inp_length) // 3
        operator[left:right] = 1
        # operator[left-1] = 10
        # operator[left - 1:left] = 5
        return torch.tensor(operator)
    else:
        raise NotImplementedError

def get_prefixed_operator_kernel(typ, kernel_size):
    if typ == "begin":
        operator = np.zeros(kernel_size, dtype=np.float32) - 5
        right = kernel_size // 3
        left = 0
        operator[left:right] = 5
        # operator[right] = 10
        print("operator = ", operator)
        return torch.tensor(operator)
    elif typ == "middle":
        operator = np.zeros(kernel_size, dtype=np.float32) - 5
        left = kernel_size // 3
        right = (2 * kernel_size) // 3
        operator[left:right] = 5
        # operator[right-1] = 10
        # operator[left-1] = 10
        return torch.tensor(operator)
    elif typ == "end":
        operator = np.zeros(kernel_size, dtype=np.float32) - 5
        right = kernel_size
        left = (2 * kernel_size) // 3
        operator[left:right] = 5
        # operator[left-1] = 10
        # operator[left - 1:left] = 5
        return torch.tensor(operator)
    else:
        raise NotImplementedError
