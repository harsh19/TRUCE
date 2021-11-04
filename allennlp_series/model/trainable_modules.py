import numpy as np
import torch.nn as nn
import torch
from overrides import overrides
from typing import Dict, List, Any

from allennlp_series.model.module_utils import *
from allennlp_series.model.conv_operators import *
from allennlp_series.common.constants import *

sigm = nn.Sigmoid()
tanh = nn.Tanh()
softmax = nn.Softmax(dim=0)


class LocationKernels(nn.Module):
    def __init__(self, kernel_set="setof6"):
        super().__init__()
        assert kernel_set in ["setof6", "setof11"]
        if kernel_set == "setof6":
            means = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        else:
            assert False
            # means = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.count = len(means)
        self.means = torch.tensor(means)
        self.stds = torch.tensor([LOCATE_KERNEL_STD] * len(means))
        if torch.cuda.is_available():
            self.stds = self.stds.cuda()
            self.means = self.means.cuda()
        self.kernels = [
            torch.distributions.normal.Normal(
                loc=self.means[i].detach(), scale=self.stds[i].detach()
            )
            for i in range(len(means))
        ]
        # if torch.cuda.is_available():
        #     self.kernels = [k.cuda() for k in self.kernels]

    def forward(self, weights, inp):
        # ret = torch.zeros_like(inp)
        ret = torch.zeros((inp.size()[0], inp.size()[1] - 2))
        if torch.cuda.is_available():
            ret = ret.cuda()
        ln = ret.size()[1]
        for idx, wt in enumerate(weights):  # iterate over fixed kernels
            for pos in range(ln):  # iterate over position
                posval = torch.tensor(pos * 1.0 / ln)
                if torch.cuda.is_available():
                    posval = posval.cuda()
                ret[:, pos] += wt * self._get_kernel_value(pos=posval, idx=idx).detach()
                # detaching since we want to keep the kernels fixed
                # wt represents weight of the idx_th kernel
        return ret

    def _get_kernel_value(self, pos, idx):
        return self.kernels[idx].log_prob(pos).exp()



def create_sinusoidal_embeddings(n_pos, dim, out):
    # print("out : ", out.size()) # bs,pos,dim
    position_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
            for pos in range(n_pos)
        ]
    )
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


class TrainableLocateModule(nn.Module):
    def __init__(
        self,
        operator_name: str = "default",
        inp_length=12,
        operator_type: str = None,
        fixed_module: bool = False,
        kernel_set="setof6",
        use_position_encoding=False,
        position_encoding_dim=10,
        alternate=False
    ):
        super().__init__()
        self.fixed_module = fixed_module
        self.num_features = None  # number of output features
        self.inp_length = inp_length  # number of input features
        self.operator_name = operator_name
        self.use_position_encoding = use_position_encoding
        self.position_encoding_dim = position_encoding_dim

        if not use_position_encoding:
            self.operator_type = operator_type
            self.locate_kernels = LocationKernels(kernel_set=kernel_set)
            if operator_type is not None:
                assert False
            else:
                # self.locate_kernels_weights = nn.Parameter(torch.rand(self.locate_kernels.count))
                self.locate_kernels_weights = nn.Parameter(
                    torch.rand(self.locate_kernels.count) - 0.5
                )  # - 0.5
                # self.locate_kernels_weights = nn.Parameter(0.1*torch.rand(self.locate_kernels.count))
                # self.locate_kernels_weights = nn.Parameter(torch.rand(self.locate_kernels.count))
                # self.operator = nn.Parameter(torch.rand(inp_length))
        else:
            # https://github.com/huggingface/transformers/blob/c483803d1b0ba9e8afa4a0bfbdb8603a892e46d5/src/transformers/modeling_xlm.py
            assert False

        self.alternate = alternate
        if alternate:
            self.baseline = nn.Parameter(torch.rand(1))

        print("TrainableLocateModule params:")
        for param in self.parameters():
            print("TrainableLocateModule: param = ", param)


    def forward(self, values: torch.Tensor):
        bs = values.size()[0]

        if not self.use_position_encoding:

            # alternate
            if not self.alternate:
                weights = 0.25 * tanh(self.locate_kernels_weights)  # b/w -1 and 1
                # empirically 1/4 worked better than 1/6 for set-of-6 case;
                # this does let the model deviate from values be higher than 1 though
                out = self.locate_kernels.forward(weights=weights, inp=values)  # out: bs,ln
                out = tanh(out)  # try sigmoid instead ?
                if self.fixed_module:
                    out = out.detach()
                # print("[Locate]: out= ", out)
                return out  # b/w -1 and 1
            else:
                weights = (1.0 + tanh(self.locate_kernels_weights) ) / 2.0  # b/w 0 and 1
                out = self.locate_kernels.forward(weights=weights, inp=values)  # out: bs,ln
                out = sigm(out - self.baseline)  # subtract a baseline
                if self.fixed_module:
                    out = out.detach()
                return out  # b/w -1 and 1

        else:
            assert False


    def get_useful_partitions(self):
        ret = {}
        name = self.operator_name
        if not self.use_position_encoding:
            operator_np = list(self.locate_kernels_weights.data.cpu().numpy())
            vals = operator_np  # sum(operator_np[:m1]), sum(operator_np[m1 - 1:m2]), sum(operator_np[m2 - 1:])
            for j in range(len(vals)):
                ret[name + "_partition" + str(j)] = float(vals[j])
            return ret
        else:
            return ret


class TrainableAttendModule(nn.Module):
    def __init__(
        self,
        operator_name: str = "default",
        inp_length=12,
        operator_init_type="none",
        learnable_bias: bool = False,
        fixed_module=False,
    ):

        super().__init__()

        self.num_features = None  # number of output features
        self.inp_length = inp_length  # number of input features
        self.operator_name = operator_name
        self.learnable_bias = learnable_bias

        # fixed_kernel_dct = {'a1':[-10.,10,0], 'a2':[0,10.,-10]}
        # provided_kernel_dct = {} #{'a1':[-0.5,0.5,0], 'a2':[0,0.5,-0.5]}
        provided_kernel_dct = {
            "peak": [-0.25, 0.5, 25],
            "increase": [-0.5, 0.5, 0.0],
            "decrease": [0.5, -0.5, 0],
            "none": None,
            "trough": [0.25, -0.5, 0.25],
        }
        provided_kernel = provided_kernel_dct[
            operator_init_type
        ]  # .get(operator_init_type,None)
        self.layer0 = SimpleConvOperator(
            inp_length=inp_length,
            operator_name=operator_name + "." + "simple_conv",
            provided_kernel=provided_kernel,
            fixed_module=False,  # False #True
        )  # -> bs,inp_length-2
        if self.learnable_bias:
            self.layer1 = nn.Sequential(nn.Linear(1, 1, bias=True), nn.Tanh())
        else:
            self.layer1 = nn.Tanh()
        self.conv_layers = [self.layer0]  # , self.layer2]
        self.out_feature_size = 10
        self.fixed_module = fixed_module

    def forward(
        self, values: torch.Tensor, deb=False, l2_mode=False, analysis_mode=False
    ):
        sz = values.size()[1]
        if deb:
            print("[TrainableAttendModule] values = ", values)
        ret = self.layer0(values.unsqueeze(1))  # ret: bs=4,1,length=8
        retl2 = torch.mean(torch.square(ret))
        if deb:
            print("[TrainableAttendModule] self.layer0(values.unsqueeze(1)) = ", ret)
            # this is getting exploded to very large values
        if analysis_mode:
            v2pre = ret
        ret = ret.squeeze(1)  # bs,length
        if self.learnable_bias:
            # print("input : ", ret.view(-1,1).size(), " || layer1 = ", self.layer1)
            bs = ret.size()[0]
            ret = self.layer1(ret.view(-1, 1))
            ret = ret.view(bs, -1)
        else:
            ret = self.layer1(ret - PATT_MODULE_LAYER1_BIAS)  # ret: bs=4,1,length=8
        ## make this bias term trainable ??
        ret = 0.5 + 0.5 * ret  # tanh output is -1 to 1. reduce that to 0-1
        if self.fixed_module:
            ret = ret.detach()
        if deb:
            print(
                "[TrainableAttendModule] operator_name = ",
                self.operator_name,
                "ret = ",
                ret,
            )
            # print(ret>=0)
        if l2_mode:
            ret = ret, retl2
        if analysis_mode:
            return {"v2": ret, "v2pre": v2pre}
        return ret

    def show_params(self):
        for j, layer in enumerate(self.conv_layers):
            print("layer {} : {} ".format(j, layer.__str__()))
            print("conv weights = ", layer.conv_weights)
            print("bias = ", layer.bias)

    def get_useful_partitions(self):
        ret = {}
        for layer in self.conv_layers:
            ret.update(layer.get_useful_partitions())
        return ret


class TrainableTwoLayerAttendModule(nn.Module):
    def __init__(
        self,
        operator_name: str = "default",
        inp_length=12,
        operator_init_type="none",
        fixed_module=False,
        learnable_bias: bool = False,
        debug=False,
    ):

        super().__init__()

        self.num_features = None  # number of output features
        self.inp_length = inp_length  # number of input features
        self.operator_name = operator_name
        # self.operator_type = operator_type
        self.learnable_bias = learnable_bias

        # provided_kernel_dct = {'peak':[-0.5,1.0,-0.5],
        #                        'increase':[-0.5,0.25,0.25],
        #                        'decrease': [0.25, 0.25, -0.5],
        #                        'trough': [0.5, -1., 0.5],
        #                        'none':None}
        # provided_kernel_dct = {'peak': [-0.25, 0.6, -0.25],
        #                        'increase': [-0.5, 0.25, 0.25],
        #                        'decrease': [0.25, 0.25, -0.5],
        #                        'trough': [0.5, -1., 0.5],
        #                        'none': None}
        provided_kernel_dct = {
            "peak": [-0.25, 0.25, 0.0],
            "increase": [-0.25, 0.20, 0.05],
            "decrease": [0.20, 0.05, -0.25],
            "trough": [0.25, -0.25, 0.0],
            "none": None,
        }
        provided_kernel = provided_kernel_dct.get(operator_init_type, None)
        self.layer0 = SimpleConvOperator(
            inp_length=inp_length,
            operator_name=operator_name + "_simple_conv",
            provided_kernel=provided_kernel,
            fixed_module=False,  # False #True
        )  # -> bs,inp_length-2
        if self.learnable_bias:
            # self.layer1 = nn.Linear(1,1,bias=True)
            self.layer1 = nn.Sequential(nn.Linear(1, 1, bias=True), nn.Tanh())
        else:
            self.layer1 = nn.Tanh()
        # self.layer1 = nn.Tanh()

        # provided_kernel_dct = {'peak':[0.5,0.5],
        #                        'increase':[0.5,0.5],
        #                        'decrease':[0.5,0.5],
        #                        'trough':[0.5,0.5]}
        provided_kernel_dct = {
            "peak": [0.25, -0.25],
            "increase": [0.25, 0.25],
            "decrease": [0.25, 0.25],
            "trough": [0.25, -0.25],
        }
        provided_kernel = provided_kernel_dct.get(operator_init_type, None)
        self.layer2 = SimpleConvOperator(
            inp_length=inp_length - 2,
            operator_name=operator_name + "_simple_conv2",
            kernel_size=2,
            provided_kernel=provided_kernel,
            # provided_bias=[-0.9],
            fixed_module=False,
            pad_len=1,  # False #True
        )  # -> bs,inp_length-3
        self.layer3 = nn.Tanh()
        self.conv_layers = [self.layer0, self.layer2]  # , self.layer2]
        # if operator_type is None:
        self.out_feature_size = 9
        # else:
        # self.out_feature_size=12
        self.fixed_module = fixed_module

    def forward(self, values: torch.Tensor, deb: bool = False):
        sz = values.size()[1]
        if deb:
            print("[TrainableTwoLayerAttendModule] values = ", values)
        ret = self.layer0(values.unsqueeze(1), deb=deb)  # ret: bs=4,1,length=10
        ret = self.layer1(ret)  # ret: bs=4,1,length=10
        ret = ret.squeeze(1)  # bs,length
        #
        ret = self.layer2(ret.unsqueeze(1), deb=deb)  # ret: bs=4,1,length=9
        ret = self.layer3(ret)  # ret: bs=4,1,length=9
        ret = ret.squeeze(1)  # bs,length
        #
        ret = 0.5 + 0.5 * ret  # tanh output is -1 to 1. reduce that to 0-1
        # print("ret after redc =", ret.size())
        # print()
        if self.fixed_module:
            ret = ret.detach()
        if deb:
            print("[TrainableTwoLayerAttendModule] ret ===> ", ret)
        return ret

    def show_params(self):
        for j, layer in enumerate(self.conv_layers):
            print("layer {} : {} ".format(j, layer.__str__()))
            print("conv weights = ", layer.conv_weights)
            print("bias = ", layer.bias)

    def get_useful_partitions(self):
        ret = {}
        for layer in self.conv_layers:
            ret.update(layer.get_useful_partitions())
        return ret


class TrainableCombineModule(nn.Module):
    def __init__(
        self,
        operator_name: str = "default",
        inp_length=12,
        use_new_defn=True,
        operator_type: str = None,
        learnable_bias: bool = False,
        log_mode: bool = False,
    ):
        super().__init__()
        self.num_features = None  # number of output features
        self.inp_length = inp_length  # number of input features
        self.operator_name = operator_name
        self.operator_type = operator_type
        self.log_mode = log_mode
        self.learnable_bias = learnable_bias
        if self.learnable_bias:
            self.layer1 = nn.Linear(1, 1, bias=True)
        assert operator_type in ["combine_exists", "combine_overlap"]
        assert use_new_defn

    def forward(self, values1: torch.Tensor, values2: torch.Tensor, deb=False):
        # print()
        if deb:
            print("[CombineModule]: values1 = ", values1)
            print("[CombineModule]: values2 = ", values2)
        # sz = values1.size()[1]
        # print()

        if self.operator_type == "combine_exists":

            if self.log_mode:
                assert False
                # tmp = torch.log(values1 + 0.501) + torch.log(values2 + 0.001)
                # tmp = values1 + torch.log(values2)
            else:
                tmp = values1 * values2
            if deb:
                print("[CombineModule] combine_exists : tmp) = ", tmp)

            if self.learnable_bias:
                # assert False
                # print("tmp : ", tmp.size())
                bs = tmp.size()[0]
                # ret = BASE_SCALE * self.layer1(tmp.view(-1, 1)).view(bs, -1)
                ret = self.layer1(tmp.view(-1, 1)).view(bs, -1)
            else:
                ret = BASE_SCALE * (tmp - BASE_ADJUST)
            if self.log_mode:
                assert False
                # pass
            else:
                ret = sigm(ret)
            if deb:
                print(
                    "[CombineModule] combine_exists : sigm(5* (tmp - sigm(0.) ) ) = ",
                    ret,
                )

            ret = torch.sum(ret, dim=1)
            if deb:
                print("[CombineModule] combine_exists : torch.sum(ret, dim=1) = ", ret)

            # ret = sigm( 2*(ret - 1.0 ) )
            if self.log_mode:
                assert False
                # score = ret
                # ret = torch.exp(score)
            else:
                score = 2 * (ret - 1.0) # real numbers in some range ; # add some bias terms ?
                ret = sigm(score) # 0 and 1

            if deb:
                print("[CombineModule] Returning ret, score =  : ", ret, score)
                print()

            return ret, score

        elif self.operator_type == "combine_overlap":
            assert False
            ## can some of these be replaced via some avergae pooling operations ?
            # tmp1: torch.Tensor = 1.0 - (1.0 - values1) * values2
            # # print(tmp1)
            # tmp1 = capped_relu(tmp1, thresh=0.2)
            # # print(tmp1)
            # tmp2: torch.Tensor = 1.0 - (1.0 - values2) * values1
            # # print(tmp2)
            # tmp2 = capped_relu(tmp2, thresh=0.2)
            # # print(tmp2)
            # # print(tmp1 * tmp2)
            # return torch.mean(tmp1 * tmp2), torch.mean(tmp1 * tmp2) > 0.2












if __name__ == "__main__":

    import pickle
    from allennlp_series.common.constants import *

    # label_data_subset_type = 'type3'
    label_data_subset_type = "all_but_throughout"

    # a = torch.tensor( [
    #     [ 7., 11., 15., 19, 23, 27, 31, 35, 39, 43, 47, 51],
    #     [77., 71., 65., 59, 53, 47, 41, 35, 29, 23, 22, 21],
    #     [7., 11., 15., 19, 23, 17, 15, 14, 13, 13, 11, 10],
    a = torch.tensor(
        [
            [
                13.0,
                21.0,
                29.0,
                37.0,
                35.0,
                35.0,
                37.0,
                38.0,
                37.0,
                35.0,
                37.0,
                35.0,
            ],  # inc-beg
            [
                16.0,
                16.0,
                17.0,
                16.0,
                22.0,
                25.0,
                29.0,
                37.0,
                37.0,
                35.0,
                37.0,
                35.0,
            ],  # inc-middle
            [
                14.0,
                14.0,
                15.0,
                14.0,
                12.0,
                16.0,
                13.0,
                14.0,
                25.0,
                36.0,
                47.0,
                45.0,
            ],  #  inc-end
            [
                33.0,
                47.0,
                75.0,
                61.0,
                19.0,
                5.0,
                6.0,
                3.0,
                6.0,
                7.0,
                6.0,
                4.0,
            ],  # peak-beg
            [
                79.0,
                40.0,
                14.0,
                27.0,
                53.0,
                66.0,
                68.0,
                68.0,
                66.0,
                64.0,
                68.0,
                68.0,
            ],  # trough-beg
            [66.0, 51.0, 36.0, 21.0, 6.0, 6.0, 5.0, 4.0, 7.0, 4.0, 5.0, 5.0],  # dec-beg
        ]
    )
    inp_length = a.size()[1]

    # increases = TrainableAttendModule(inp_length, operator_type='increase')
    # increases = TrainableAttendModule(inp_length=inp_length, operator_name='dasjknak')
    # increases = TrainableAttendModule(inp_length=inp_length, operator_name='a1')

    begin = TrainableLocateModule(
        operator_type="begin",
        operator_name="begin",
        inp_length=inp_length,
        fixed_module=False,
        kernel_set="setof6",
    )
    middle = TrainableLocateModule(
        operator_type="middle",
        inp_length=inp_length,
        fixed_module=False,
        kernel_set="setof6",
    )
    end = TrainableLocateModule(
        operator_type="end",
        inp_length=inp_length,
        fixed_module=False,
        kernel_set="setof6",
    )

    # middle = TrainableLocateModule(operator_type='middle', inp_length=inp_length-3, fixed_module=True)
    combine_exists = TrainableCombineModule(
        operator_type="combine_exists", inp_length=inp_length
    )
    # peaks = TrainableTwoLayerAttendModule(inp_length=inp_length, operator_name='peak')
    # inctwo = TrainableTwoLayerAttendModule(inp_length=inp_length, operator_name='inc')

    peaks = TrainableTwoLayerAttendModule(
        inp_length=inp_length, operator_name="peak", operator_init_type="peak"
    )
    increase = TrainableTwoLayerAttendModule(
        inp_length=inp_length, operator_name="increase", operator_init_type="increase"
    )
    decrease = TrainableTwoLayerAttendModule(
        inp_length=inp_length, operator_name="decrease", operator_init_type="decrease"
    )
    trough = TrainableTwoLayerAttendModule(
        inp_length=inp_length, operator_name="trough", operator_init_type="trough"
    )

    # peaks = TrainableAttendModule(inp_length=inp_length, operator_name='peak', operator_init_type='peak')
    # increase = TrainableAttendModule(inp_length=inp_length, operator_name='increase', operator_init_type='increase')
    # decrease = TrainableAttendModule(inp_length=inp_length, operator_name='decrease', operator_init_type='decrease')
    # trough = TrainableAttendModule(inp_length=inp_length, operator_name='trough', operator_init_type='trough')

    print("=" * 41)
    v1 = begin(a)
    print("begin: ", v1)

    v1 = middle(a)
    print("middle: ", v1)

    v1 = end(a)
    print("end: ", v1)
    print("=" * 41)

    # v2 = increases(a)
    # print("increases: ",v2)

    # v2inc = inctwo(a)
    # print("inctwo increases: ", v2inc)
    #
    # val = peaks(a, deb=True)
    # print("peaks: ",val)
    # print("="*41)

    v2 = increase(a)  # , deb=True)
    print("increase: ", v2)
    print("=" * 41)

    # v2 = decrease(a, deb=True)
    # print("decrease: ",v2)
    # print("="*41)

    # val = trough(a, deb=True)
    # print("trough: ",val)
    # print("="*41)

    # v3 = combine(v1,v2)
    # print("combine: ",v3)

    v3 = combine_exists(v1, v2)
    print("------->>>> combine_exists(): ", v3)

    # v3 = combine_exists(v1, v2inc)
    # print("combine_exists(middle, inc): ", v3)

    # 0/0

    file_path = "synthetic_data/newseries_set2_combined_dev.pkl"
    # file_path = 'synthetic_data/newseries_set2_combined_length30_dev.pkl'

    print("Reading instances from file at: ", file_path)
    data = pickle.load(open(file_path, "rb"))
    all_cols, all_labels = data
    sz = len(all_labels)
    print("sz = ", sz)
    inst_num = 0

    label_data_subset_type = LABEL_DATA_TYPE_SUBSET_TYPE_LIST[label_data_subset_type]
    label_list = label_data_subset_type  # self.label_data_subset_type #LABEL_DATA_TYPE_SUBSET_TYPE1
    # label_to_labeltext = {0:'increase_all',
    #     1:'incerease_begin',
    #     2:'incerease_middle',
    #     3:'incerease_end',
    #     4:'decerease_all',
    #     5:'decerease_begin',
    #     6:'decerease_middle',
    #     7:'decerease_end',
    #     8:'peak_begin',
    #     9:'peak_middle',
    #     10:'peak_end',
    #     11:'trough_begin',
    #     12:'trough_middle',
    #     13:'trough_end'
    #    }
    pos_summary = []  # np.zeros()
    pos_summary_score = []  # np.zeros()
    neg_summary = []  # np.zeros()
    neg_summary_score = []  # np.zeros()
    # poslabel = 'decerease_middle'
    # poslabel = 'peak_middle'
    poslabel = "peak_begin"
    # poslabel = 'trough_middle'
    # poslabel = 'decerease_end'
    # poslabel = 'decerease_begin'
    # poslabel = 'incerease_middle'
    # poslabel = 'incerease_middle'
    for col, label in zip(all_cols, all_labels):
        cur_label = label["labels"]
        if cur_label not in label_list:
            continue
        else:
            cur_label = label["label_text"]  # label_to_labeltext[cur_label]
            print(cur_label == poslabel, cur_label)
            # print(cur_label == 'peak_middle')
            a = torch.tensor([np.array(col, dtype=np.float32)])
            # v1 = end(a)
            # v1 = middle(a)
            v1 = begin(a)
            # print("end: ", v1)
            # print("=" * 41)
            #
            # v2 = increase(a)  # , deb=True)
            # v2 = decrease(a, deb=True)
            v2 = peaks(a, deb=True)
            # v2 = trough(a, deb=True)
            # print("increase: ", v2)
            # print("=" * 41)
            v3, v3score = combine_exists(v1, v2)
            print(cur_label)
            print("combine_exists(): ", v3)
            # if cur_label == 'incerease_end':
            if cur_label == poslabel:
                # if cur_label == 'peak_middle':
                pos_summary.append(v3.item())
                pos_summary_score.append(v3score.item())
            # elif cur_label == 'peak_middle':
            #     neg_summary.append(v3.item())
            else:
                neg_summary.append(v3.item())
                neg_summary_score.append(v3score.item())
            print()
        inst_num += 1

    import scipy.stats

    pos_summary = np.array(pos_summary)
    neg_summary = np.array(neg_summary)

    print("====pos_summary====")
    print(scipy.stats.describe(np.array(pos_summary)))

    print("====neg_summary====")
    print(scipy.stats.describe(np.array(neg_summary)))

    print("====pos_summary score====")
    print(scipy.stats.describe(np.array(pos_summary_score)))

    print("====neg_summary score====")
    print(scipy.stats.describe(np.array(neg_summary_score)))

    print("========")

    for j in range(3, 20):
        thresh = 0.05 * j
        print(
            "thresh = ",
            thresh,
            " || pos_summary(pos>thresh): ",
            sum(pos_summary > thresh) / len(pos_summary),
            " || neg_summary(neg<thresh): ",
            sum(neg_summary <= thresh) / len(neg_summary),
        )


# DECREASE END
# ====pos_summary====
# DescribeResult(nobs=250, minmax=(0.22008350491523743, 0.3454223573207855), mean=0.3336974932551384, variance=0.0007177416925215207, skewness=-2.9914098016849024, kurtosis=7.748228090810487)
# ====neg_summary====
# DescribeResult(nobs=2750, minmax=(0.1725701540708542, 0.35070809721946716), mean=0.23387509152564134, variance=0.002019988734321553, skewness=0.8997975095930073, kurtosis=0.5898100949039469)
# ====pos_summary score====
# DescribeResult(nobs=250, minmax=(-1.2651798725128174, -0.6392223834991455), mean=-0.6946305408477783, variance=0.01698875295908562, skewness=-3.063405454538542, kurtosis=8.278680422693377)
# ====neg_summary score====
# DescribeResult(nobs=2750, minmax=(-1.5675204992294312, -0.6159281730651855), mean=-1.2020771193504334, variance=0.05938419533275195, skewness=0.586477419114792, kurtosis=0.048037763732380334)
# ========
# thresh =  0.15000000000000002  || pos_summary(pos>thresh):  1.0  || neg_summary(neg<thresh):  0.0
# thresh =  0.2  || pos_summary(pos>thresh):  1.0  || neg_summary(neg<thresh):  0.23236363636363636
# ** # thresh =  0.25  || pos_summary(pos>thresh):  0.96  || neg_summary(neg<thresh):  0.7010909090909091
# thresh =  0.30000000000000004  || pos_summary(pos>thresh):  0.9  || neg_summary(neg<thresh):  0.9232727272727272
# thresh =  0.35000000000000003  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  0.9970909090909091
# thresh =  0.4  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# thresh =  0.45  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# thresh =  0.5  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# thresh =  0.55  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# thresh =  0.6000000000000001  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# thresh =  0.65  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# thresh =  0.7000000000000001  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# thresh =  0.75  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# thresh =  0.8  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# thresh =  0.8500000000000001  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# thresh =  0.9  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# thresh =  0.9500000000000001  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# (torchnew) jharsh@banyan:~/projects/time_series_captioning$


#
# TROUGH MIDDLE
# ====pos_summary====
# DescribeResult(nobs=250, minmax=(0.29218870401382446, 0.4030372202396393), mean=0.37359327912330625, variance=0.0009207170407286583, skewness=-1.545223020846746, kurtosis=2.1029509517284675)
# ====neg_summary====
# DescribeResult(nobs=2750, minmax=(0.24253599345684052, 0.36445897817611694), mean=0.29845622450655157, variance=0.000923731637826835, skewness=0.0413151973443951, kurtosis=-0.9707959266725035)
# ====pos_summary score====
# DescribeResult(nobs=250, minmax=(-0.8847776651382446, -0.3928258419036865), mean=-0.519340913772583, variance=0.018018394831874313, skewness=-1.6440975377718505, kurtosis=2.357528954124539)
# ====neg_summary score====
# DescribeResult(nobs=2750, minmax=(-1.13882577419281, -0.5560626983642578), mean=-0.8589425986896861, variance=0.02126364213217571, skewness=-0.049434711252741986, kurtosis=-0.9672843981036046)
# ========
# thresh =  0.15000000000000002  || pos_summary(pos>thresh):  1.0  || neg_summary(neg<thresh):  0.0
# thresh =  0.2  || pos_summary(pos>thresh):  1.0  || neg_summary(neg<thresh):  0.0
# thresh =  0.25  || pos_summary(pos>thresh):  1.0  || neg_summary(neg<thresh):  0.04363636363636364
# thresh =  0.30000000000000004  || pos_summary(pos>thresh):  0.904  || neg_summary(neg<thresh):  0.5247272727272727
# ** # thresh =  0.35000000000000003  || pos_summary(pos>thresh):  0.904  || neg_summary(neg<thresh):  0.96
# thresh =  0.4  || pos_summary(pos>thresh):  0.328  || neg_summary(neg<thresh):  1.0
# thresh =  0.45  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# thresh =  0.5  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# thresh =  0.55  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# thresh =  0.6000000000000001  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# thresh =  0.65  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# thresh =  0.7000000000000001  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# thresh =  0.75  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# thresh =  0.8  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# thresh =  0.8500000000000001  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# thresh =  0.9  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# thresh =  0.9500000000000001  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0


# PEAK BEGIN
# thresh =  0.15000000000000002  || pos_summary(pos>thresh):  1.0  || neg_summary(neg<thresh):  0.0
# thresh =  0.2  || pos_summary(pos>thresh):  1.0  || neg_summary(neg<thresh):  0.0
# thresh =  0.25  || pos_summary(pos>thresh):  1.0  || neg_summary(neg<thresh):  0.042545454545454546
# thresh =  0.30000000000000004  || pos_summary(pos>thresh):  0.928  || neg_summary(neg<thresh):  0.488
# ** # thresh =  0.35000000000000003  || pos_summary(pos>thresh):  0.928  || neg_summary(neg<thresh):  0.936
# thresh =  0.4  || pos_summary(pos>thresh):  0.424  || neg_summary(neg<thresh):  1.0
# thresh =  0.45  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# thresh =  0.5  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# thresh =  0.55  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# thresh =  0.6000000000000001  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# thresh =  0.65  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# thresh =  0.7000000000000001  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# thresh =  0.75  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# thresh =  0.8  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# thresh =  0.8500000000000001  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# thresh =  0.9  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0
# thresh =  0.9500000000000001  || pos_summary(pos>thresh):  0.0  || neg_summary(neg<thresh):  1.0


# analysis
# TrainableCombineModule
# with values from metrics.json
# model_name = 'stock_klanneal_bs32_heurcfactlabels_exp1a_seed11'
# tmp/model_name/metrics.json
# metrics = json.load(open('../tmp/'+model_name+'/metrics.json','r'))
#
