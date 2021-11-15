import numpy as np
import torch.nn as nn
import torch
from overrides import overrides
from typing import Dict, List, Any

# from allennlp_series.model.model_archive.operations_configs import *
from allennlp_series.model.module_utils import *

sigm = nn.Sigmoid()
#
#
#
# class LocationKernels(nn.Module):
#
#     def __init__(self ):
#         super().__init__()
#         means = [0.0,0.2,0.4,0.6,0.8,1.0]
#         self.count = len(means)
#         self.means = torch.tensor( means )
#         self.stds = torch.tensor( [0.1]*len(means) )
#         self.kernels = [ torch.distributions.normal.Normal(loc=self.means[i].detach(),
#                                                            scale=self.stds[i].detach()) for i in range(len(means)) ]
#
#     def forward(self, weights, inp):
#         ret = torch.zeros_like(inp)
#         ln = ret.size()[1]
#         for idx,wt in enumerate(weights): # iterate over kernels
#             for pos in range(ln): # iterate over position
#                 ret[:,pos] += wt * self._get_kernel_value(pos=pos*1.0/ln,idx=idx).detach()
#                 # detaching since we awant to keep the kernels fixed
#                 # wt represents weight of the ith kernel
#         return ret
#
#     def _get_kernel_value(self,pos,idx):
#         return self.kernels[idx].log_prob(pos).exp()

# locate_kernels = [ LocationKernel(mean,std=1.0) for mean in [0.0,0.2,0.4,0.6,0.8,1.0] ]


# class LocateModule(nn.Module):
#
#     def __init__(self, operator_name: str = 'default', inp_length=12, operator_type:str=None, fixed_module:bool=False):
#         super().__init__()
#         self.fixed_module = fixed_module
#         self.num_features = None  # number of output features
#         self.inp_length = inp_length  # number of input features
#         self.operator_name = operator_name
#         self.operator_type = operator_type
#         assert operator_type in ['middle','begin','end','all']
#         if operator_type is not None:
#             self.operator = nn.Parameter( get_prefixed_operator(typ=operator_type, inp_length=inp_length) )
#         else:
#             self.operator = nn.Parameter(torch.rand(inp_length))
#
#     def forward(self, values: torch.Tensor):
#         bs = values.size()[0]
#         if self.fixed_module:
#             return self.operator.unsqueeze(0).repeat(bs,1).detach()
#         else:
#             return self.operator.unsqueeze(0).repeat(bs,1)
#
# class LocateModule(nn.Module):
#
#     def __init__(self, operator_name: str = 'default', inp_length=12, operator_type: str = None,
#                  fixed_module: bool = False):
#         super().__init__()
#         self.fixed_module = fixed_module
#         self.num_features = None  # number of output features
#         self.inp_length = inp_length  # number of input features
#         self.operator_name = operator_name
#         self.operator_type = operator_type
#         assert operator_type in ['middle', 'begin', 'end', 'all']
#         self.locate_kernels = LocationKernels()
#         if operator_type is not None:
#             # raise NotImplementedError
#             # self.operator = nn.Parameter(get_prefixed_operator(typ=operator_type, inp_length=inp_length))
#             self.locate_kernels_weights = nn.Parameter(get_prefixed_operator_kernel(typ=operator_type,
#                                                        kernel_size=self.locate_kernels.count))
#         else:
#             self.locate_kernels_weights = nn.Parameter(torch.rand(self.locate_kernels.count))
#             # self.operator = nn.Parameter(torch.rand(inp_length))
#
#     def forward(self, values: torch.Tensor):
#         bs = values.size()[0]
#         weights = self.locate_kernels_weights
#         # todo - run through tanh / sigmoid ?
#         # print("weights = ", weights)
#         out = self.locate_kernels.forward(weights=weights,inp=values) # out: bs,ln
#         if self.fixed_module:
#             out = out.detach()
#         # print("[Locate]: out= ", out)
#         return out
#
#     '''
#     - new locate
#     - consider fixed kernels at known positions - 0,0.2,0.4,0.6,..,1.0
#     - each module gets its weight [w1,w2,..,w6]
#     - in forward,
#         - init with a = [0,0,...,0] ->length = input length = 12 lets say
#         - each kernel represents mean at len(=12)*kernel's position
#         - so construct a = w1*kernel1 + w2*kernel2 + ...
#         - can also keep std of each kernel as learnable
#     '''
#
#


class AttendModule(nn.Module):
    def __init__(
        self, operator_name: str = "default", inp_length=12, operator_type: str = None
    ):
        super().__init__()
        self.num_features = None  # number of output features
        self.inp_length = inp_length  # number of input features
        self.operator_name = operator_name
        self.operator_type = operator_type
        assert operator_type in ["increase", "decrease", "peak", "trough"]

    def forward(self, values: torch.Tensor):
        sz = values.size()[1]
        ret = torch.zeros_like(values)
        if self.operator_type == "increase":
            # print("increases: input = ", values)
            for j in range(sz - 1):
                ret[:, j] = (
                    values[:, j + 1] - values[:, j]
                )  ## this can be thought of as 1-d convolution
            # print("increases: step 1: ", ret)
            ret = ret * 0.1  # Replace with affine transformation --->
            # print("increases step2: ", ret)
            ret = capped_relu(ret, 0.2)
            # print("increases step2: ", ret)
            ret[:, sz - 1] = ret[:, sz - 2]  # padding
            return ret
        elif self.operator_type == "decrease":
            # print("decrease: input = ", values)
            for j in range(sz - 1):
                ret[:, j] = values[:, j] - values[:, j + 1]
            # print("decrease step1: ", ret)
            ret = ret * 0.1  # Replace with affine transformation
            # print("decrease step2: ", ret)
            ret = capped_relu(ret, 0.2)
            # print("decrease step3: ", ret)
            ret[:, sz - 1] = ret[:, sz - 2]  # padding
            return ret
        elif self.operator_type == "peak":
            # print("peak: input = ", values)
            ret1 = torch.zeros_like(values)
            ret2 = torch.zeros_like(values)
            wt1 = torch.tensor([0.3, 0.7]).unsqueeze(0)  # 1,2
            wt2 = torch.tensor([0.7, 0.3]).unsqueeze(0)  # 1,2
            for j in range(sz - 1):
                ret1[:, j] = 0.1 * (values[:, j + 1] - values[:, j])  # inc.
            for j in range(sz - 1):
                ret2[:, j] = 0.1 * (values[:, j] - values[:, j + 1])  # dec.
            ret1 = capped_relu(ret1, 0.2)
            ret2 = capped_relu(ret2, 0.2)
            # print("peak step2: ret1 ", ret1)
            # print("peak step2: ret2 ", ret2)
            for j in range(2, sz - 2):
                print(j)
                print(
                    "wt1 * ret1[j-2:j] = ",
                    wt1,
                    ret1[:, j - 2 : j],
                    wt1 * ret1[:, j - 2 : j],
                )
                print(
                    "wt1 * ret1[j-2:j] = ", torch.sum(wt1 * ret1[:, j - 2 : j], dim=1)
                )
                print(
                    "wt2 * ret2 = ", wt2, ret2[:, j : j + 2], wt2 * ret1[:, j : j + 2]
                )
                print("wt2 * ret2 = ", torch.sum(wt2 * ret2[:, j : j + 2], dim=1))
                ret[:, j] = torch.sqrt(
                    torch.sum(wt1 * ret1[:, j - 2 : j], dim=1)
                    * 0.5
                    * torch.sum(wt2 * ret2[:, j : j + 2], dim=1)
                )
                # incr. from j-2->j-1 and j-1->j i.e. ret[j-2] and ret[j-1]
                # dec. from j->j+1 and j+1->j+2 i.e. ret2[j] and ret2[j+1]
            # print("peak step3: ", ret)
            ret = capped_relu(ret * 2.0, 0.0)
            # print("peak step4: ", ret)
            return ret
        elif self.operator_type == "trough":
            # print("trough: input = ", values)
            ret1 = torch.zeros_like(values)
            ret2 = torch.zeros_like(values)
            wt1 = torch.tensor([0.3, 0.7]).unsqueeze(0)  # 1,2
            wt2 = torch.tensor([0.7, 0.3]).unsqueeze(0)  # 1,2
            for j in range(sz - 1):
                ret2[:, j] = 0.1 * (values[:, j + 1] - values[:, j])  # inc.
            for j in range(sz - 1):
                ret1[:, j] = 0.1 * (values[:, j] - values[:, j + 1])  # dec.
            ret1 = capped_relu(ret1, 0.2)
            ret2 = capped_relu(ret2, 0.2)
            # print("trough step2: ret1 ", ret1)
            # print("trough step2: ret2 ", ret2)
            for j in range(2, sz - 2):
                # print(j)
                # print("wt1 * ret1[j-2:j] = ", wt1, ret1[:, j - 2:j], wt1 * ret1[:, j - 2:j])
                # print("wt1 * ret1[j-2:j] = ", torch.sum(wt1 * ret1[:, j - 2:j], dim=1))
                # print("wt2 * ret2 = ", wt2, ret2[:, j:j + 2], wt2 * ret1[:, j:j + 2])
                # print("wt2 * ret2 = ", torch.sum(wt2 * ret2[:, j:j + 2], dim=1))
                ret[:, j] = torch.sqrt(
                    torch.sum(wt1 * ret1[:, j - 2 : j], dim=1)
                    * 0.5
                    * torch.sum(wt2 * ret2[:, j : j + 2], dim=1)
                )
                # incr. from j-2->j-1 and j-1->j i.e. ret[j-2] and ret[j-1]
                # dec. from j->j+1 and j+1->j+2 i.e. ret2[j] and ret2[j+1]
            # print("trough step3: ", ret)
            ret = capped_relu(ret * 2.0, 0.0)
            # print("trough step4: ", ret)
            return ret


#
# class CombineModule(nn.Module):
#
#     def __init__(self, operator_name: str = 'default', inp_length=12, operator_type:str=None, use_new_defn:bool=False):
#         super().__init__()
#         self.num_features = None  # number of output features
#         self.inp_length = inp_length  # number of input features
#         self.operator_name = operator_name
#         self.operator_type = operator_type
#         assert operator_type in ['combine_exists','combine_overlap']
#         self.use_new_defn = use_new_defn
#
#     def forward(self, values1: torch.Tensor, values2: torch.Tensor):
#
#         # print("[CombineModule]: values1 = ", values1)
#         # print("[CombineModule]: values2 = ", values2)
#
#         if self.use_new_defn:
#
#             if self.operator_type == 'combine_exists':
#
#                 # print("values1 : ", values1.size())
#                 # print("values2 : ", values2.size())
#                 tmp = values1 * values2
#                 # print("[CombineModule] combine_exists : tmp) = ", tmp)
#
#                 ret = sigm(10 * (tmp - 0.3775 ) )
#                 # print("[CombineModule] combine_exists : sigm(5* (tmp - sigm(0.) ) ) = ", ret)
#
#                 ret = torch.sum(ret, dim=1)
#                 # print("[CombineModule] combine_exists : torch.sum(ret, dim=1) = ", ret)
#
#                 score = 2*(ret - 1.0 )
#                 ret = sigm( score )
#                 # print("[CombineModule] Returning sigm( ret - 1.0) : ", ret)
#
#                 return ret, score
#
#         else:
#
#             # sz = values1.size()[1]
#             if self.operator_type == 'combine_exists':
#                 tmp = values1 * values2
#                 ret = capped_relu(tmp, 0.1)
#                 # print("combine_exists : capped_relu(tmp, 0.1) = ", ret)
#                 tmp = capped_relu(torch.sum(ret,dim=1),0.5)+0.01
#                 # print("[CombineModule] Returning : ", tmp)
#                 return tmp, None
#             elif self.operator_type == 'combine_overlap':
#                 ## can some of these be replaced via some avergae pooling operations ?
#                 tmp1:torch.Tensor = 1.0 - (1.0 - values1) * values2
#                 # print(tmp1)
#                 tmp1 = capped_relu(tmp1, thresh=0.2)
#                 # print(tmp1)
#                 tmp2:torch.Tensor = 1.0 - (1.0 - values2) * values1
#                 # print(tmp2)
#                 tmp2 = capped_relu(tmp2, thresh=0.2)
#                 # print(tmp2)
#                 # print(tmp1 * tmp2)
#                 return torch.mean(tmp1 * tmp2), None #torch.mean(tmp1 * tmp2) > 0.2


class SimpleProgramType(nn.Module):
    def __init__(self, defn, instances: Dict[str, List[Any]]):
        super().__init__()
        self.locate: LocateModule = instances["locate"][defn["locate"]]  # ['instance']
        self.attend: AttendModule = instances["attend"][defn["attend"]]  # ['instance']
        self.combine: CombineModule = instances["combine"][
            defn["combine"]
        ]  # ['instance']
        self.defn = defn

    @overrides
    def forward(
        self,
        series: torch.Tensor,
        get_score_also: bool = False,
        l2_mode: bool = False,
        analysis_mode: bool = False,
        deb: bool = False,
    ):
        # inp = None

        v1 = self.locate(series)
        # print("locate: ", self.locate.operator_name, v1)

        v2 = self.attend(series, l2_mode=l2_mode, analysis_mode=analysis_mode, deb=deb)
        # print("self.attend = ", self.attend)
        # print("attend: ", self.attend.operator_name , v2)
        if analysis_mode:
            # print("v2 : ", type(v2))
            # print(v2.keys())
            v2pre = v2["v2pre"]
            v2 = v2["v2"]
        if l2_mode:
            v2, v2l2 = v2

        v3, score = self.combine(v1, v2, deb=deb)
        # print("combine_exists: ", v3)

        ret = v3
        if get_score_also:
            ret = v3, score
            if l2_mode:
                ret = v3, score, v2l2

        if analysis_mode:
            return {
                "locate": v1,
                "attend": v2,
                "attendpre": v2pre,
                "combine_v3": v3,
                "combine_score": score,
                "ret": ret,
            }

        return ret

    @property
    def __str__(self):
        tmp = "[SimpleProgramType]: self.locate={} self.attend ={} self.combine ={}".format(
            self.locate.operator_name,
            self.attend.operator_name,
            self.combine.operator_name,
        )
        return tmp


if __name__ == "__main__":

    # a = torch.tensor( [[ 7., 11., 15., 19, 23, 27, 31, 35, 39, 43, 47, 51]] )
    a = torch.tensor(
        [
            [7.0, 11.0, 15.0, 19, 23, 20, 18, 15, 9, 8, 7, 5],
            [7.0, 11.0, 15.0, 19, 23, 27, 31, 35, 39, 43, 47, 51],
            [81, 77.0, 71.0, 65.0, 59, 63, 67, 71, 75, 79, 83, 87],
        ]
    )

    middle = LocateModule(operator_type="middle")
    begin = LocateModule(operator_type="begin")
    # peaks = AttendModule(operator_type='increase')
    increases = AttendModule(operator_type="increase")
    # peaks = AttendModule(operator_type='peak')
    # peaks = AttendModule(operator_type='trough')
    combine_exists = CombineModule(operator_type="combine_exists")

    v1 = middle(a)
    print("middle: ", v1)

    v2 = increases(a)
    # v2 = peaks(a)
    print("increases: ", v2)
    # print("peaks: ",v2)

    # v3 = combine(v1,v2)
    # print("combine: ",v3)

    v3 = combine_exists(v1, v2)
    # print("combine_exists(middle, increases): ",v3)
    print("combine_exists(middle, peaks): ", v3)

    print("==============")

    v1 = begin(a)
    print("begin: ", v1)

    v2 = increases(a)
    # v2 = peaks(a)
    print("increases: ", v2)
    # print("peaks: ", v2)

    # v3 = combine(v1,v2)
    # print("combine: ",v3)

    v3 = combine_exists(v1, v2)
    # print("combine_exists(middle, increases): ",v3)
    print("combine_exists(middle, peaks): ", v3)
