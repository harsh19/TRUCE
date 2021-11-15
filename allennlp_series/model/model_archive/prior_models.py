import json
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from allennlp_series.common.constants import *


from overrides import overrides
from typing import Tuple, Dict, List, Any


class PriorModel(nn.Module):
    def __init__(
        self,
        series_length: int = None,
        num_programs: int = None,
        typ: str = "two_layer",
        embedding_dim: int = 10,
    ):
        super().__init__()
        self.num_programs = num_programs
        self.embedding_dim = embedding_dim
        self.series_length = series_length
        self.program_embedding = nn.Embedding(self.num_programs, self.embedding_dim)
        self.layer1 = nn.Linear(self.series_length, self.num_programs)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.typ = typ

    @overrides
    def forward(
        self, series: torch.Tensor
    ) -> Dict[str, torch.Tensor]:  # -> torch.Tensor: # -> Dict[str,torch.Tensor]:
        """
        :param series:
        :return: 1) distribution
        """
        logits = self.layer1(series)  # bs, length -> bs, num_programs
        dist_programs = self.log_softmax(logits)  # bs, num_programs
        return {"dist_programs": dist_programs, "logits": logits}

    def sample(
        self, logits: torch.Tensor
    ) -> Tuple[torch.IntTensor, torch.FloatTensor, torch.FloatTensor]:
        dist: torch.distributions.Categorical = torch.distributions.Categorical(
            logits=logits
        )
        program_idx = dist.sample()  # bs
        # print("program_idx = ", program_idx)
        return (
            program_idx,
            self.program_embedding(program_idx),
            dist.log_prob(program_idx),
        )  # bs;; bs, emb_dim

    def get_layer_wise(
        self,
        program_idx: torch.Tensor,
        num_operations_per_layer: int,
        num_operations_per_layer3: int,
    ) -> Tuple[List, List]:
        # program_idx: bs,
        if self.typ == "two_layer":
            program_idx_numpy = program_idx.data.cpu().numpy()
            program_idx_layer1 = np.floor_divide(
                program_idx_numpy, num_operations_per_layer
            )
            program_idx_layer2 = (
                program_idx_numpy - num_operations_per_layer * program_idx_layer1
            )
            return list(program_idx_layer1), list(program_idx_layer2)
        elif self.typ == "three_layer":
            program_idx_numpy = program_idx.data.cpu().numpy()
            program_idx_layer1 = np.floor_divide(
                program_idx_numpy, num_operations_per_layer * num_operations_per_layer3
            )
            program_idx_numpy = program_idx_numpy % (
                num_operations_per_layer * num_operations_per_layer3
            )
            program_idx_layer2 = np.floor_divide(
                program_idx_numpy, num_operations_per_layer3
            )
            program_idx_layer3 = program_idx_numpy % num_operations_per_layer3
            return (
                list(program_idx_layer1),
                list(program_idx_layer2),
                list(program_idx_layer3),
            )
        else:
            assert False


class NewPriorModel(nn.Module):
    def __init__(
        self,
        series_length: int = None,
        typ: str = "two_layer",
        num_operations_per_layer1: int = None,
        num_operations_per_layer2: int = None,
        num_operations_per_layer3: int = None,
        joint_first_two: bool = True,
        embedding_dim: int = 10,
    ):
        super().__init__()
        # self.num_programs = num_programs
        self.embedding_dim = embedding_dim
        self.series_length = series_length
        self.program_embedding1 = nn.Embedding(
            num_operations_per_layer1, self.embedding_dim
        )
        if not joint_first_two:
            self.program_embedding2 = nn.Embedding(
                num_operations_per_layer2, self.embedding_dim
            )
        else:
            self.program_embedding2 = self.program_embedding1
        self.program_embedding3 = nn.Embedding(
            num_operations_per_layer3, self.embedding_dim
        )
        self.program_embedding = [
            self.program_embedding1,
            self.program_embedding2,
            self.program_embedding3,
        ]
        self.layer1 = nn.Linear(self.series_length, num_operations_per_layer1)
        self.layer2 = nn.Linear(self.series_length, num_operations_per_layer2)
        # ** TODO: ultimately we would want to condition lauyer3 out on layer2., layer2 out on layer1, and so on
        self.layer3 = nn.Linear(self.series_length, num_operations_per_layer3)
        self.layers = [self.layer1, self.layer2, self.layer3]
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.typ = typ

    @overrides
    def forward(self, series: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        # -> torch.Tensor: # -> Dict[str,torch.Tensor]:
        """
        :param series:
        :return: 1) distribution
        """
        logits = []
        dist_programs = []
        for layer_num in range(3):
            logits.append(
                self.layers[layer_num](series)
            )  # bs, length -> bs, num_programs
            dist_programs.append(self.log_softmax(logits[-1]))  # bs, num_programs
        return {"dist_programs": dist_programs, "logits": logits}

    def sample(
        self, logits: torch.Tensor, layer_num: int
    ) -> Tuple[torch.IntTensor, torch.FloatTensor, torch.FloatTensor]:
        dist: torch.distributions.Categorical = torch.distributions.Categorical(
            logits=logits
        )
        program_idx = dist.sample()  # bs
        # print("program_idx = ", program_idx)
        return (
            program_idx,
            self.program_embedding[layer_num](program_idx),
            dist.log_prob(program_idx),
        )  # bs;; bs, emb_dim


class SimplePriorModel(nn.Module):
    def __init__(
        self,
        series_length: int = None,
        num_programs: int = None,
        embedding_dim: int = 5,
        typ: str = "simpleprior",
        posterior_addition_dim: int = None,
        provided_emb=None,
    ):
        super().__init__()
        self.num_programs = num_programs
        self.embedding_dim = embedding_dim
        self.series_length = series_length
        self.posterior_addition_dim = posterior_addition_dim
        self.typ = typ
        if provided_emb is not None:
            self.program_embedding = provided_emb
        else:
            self.program_embedding = nn.Embedding(self.num_programs, self.embedding_dim)
        if typ == "simpleprior":
            self.layer1 = nn.Linear(self.series_length, self.num_programs)
        elif typ == "simple_posterior":
            self.layer1 = nn.Linear(
                self.series_length + posterior_addition_dim, self.num_programs
            )
        self.log_softmax = nn.LogSoftmax(dim=1)

    @overrides
    def forward(
        self, series: torch.Tensor, posterior_addition_feats: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        inp = None
        if self.typ == "simpleprior":
            inp = series
        elif self.typ == "simple_posterior":
            # print("**series:", type(series)
            # print("**series:", series.size())
            inp = torch.cat(
                [series, posterior_addition_feats], dim=1
            )  # bs, leng and bs, feats -> bs, leng+feats
        logits = self.layer1(inp)  # bs, length -> bs, num_programs
        dist_programs = self.log_softmax(logits)  # bs, num_programs
        return {"dist_programs": dist_programs, "logits": logits}

    def sample(
        self, logits: torch.Tensor
    ) -> Tuple[torch.IntTensor, torch.FloatTensor, torch.FloatTensor]:
        dist: torch.distributions.Categorical = torch.distributions.Categorical(
            logits=logits
        )
        program_idx = dist.sample()  # bs
        return (
            program_idx,
            self.program_embedding(program_idx),
            dist.log_prob(program_idx),
        )  # bs;; bs, emb_dim

    def get_prob(
        self, logits: torch.Tensor, action: torch.IntTensor
    ) -> torch.FloatTensor:
        dist: torch.distributions.Categorical = torch.distributions.Categorical(
            logits=logits
        )
        log_prob = dist.log_prob(action)  # bs
        return log_prob

    def get_argmax(
        self, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.FloatTensor, torch.FloatTensor]:
        _, program_idx = torch.max(logits, dim=1)  # logits: bs,sz
        dist: torch.distributions.Categorical = torch.distributions.Categorical(
            logits=logits
        )
        return (
            program_idx,
            self.program_embedding(program_idx),
            dist.log_prob(program_idx),
        )  # bs;; bs, emb_dim

    @property
    def __str__(self):
        return "[SimplePriorModel]: self.series_length={} self.num_programs ={}".format(
            self.series_length, self.num_programs
        )


class JoinedPriorModel(nn.Module):
    def __init__(
        self,
        series_length: int = None,
        num_layers: int = 3,
        num_programs: List[int] = None,
        prior_type: str = "simpleprior",  # simple_prior,simple_posterior
        share_embeddings: str = "-1_-1_-1",
        embedding_dim: int = 5,
        use_vertical_opers: bool = False,
        num_cols_in_multiseries: int = 2,
        vertical_num_programs: int = None,
    ):
        super().__init__()
        self.prior_network: List[SimplePriorModel] = []
        if use_vertical_opers:
            pass  # num_layers += 1 ## ??
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.use_vertical_opers = use_vertical_opers
        self.prior_type = prior_type
        self.share_embeddings = share_embeddings
        share_embeddings = share_embeddings.split("_")
        assert len(share_embeddings) == num_layers
        if use_vertical_opers:
            self.vertical_prior_network = SimplePriorModel(
                series_length=num_cols_in_multiseries * series_length,
                num_programs=vertical_num_programs,
                embedding_dim=embedding_dim,
                typ=prior_type,
            )
            # self.prior_network.append(self.vertical_prior_network)
        for j in range(num_layers):
            assert prior_type in ["simpleprior"]
            share_embedding_j = int(share_embeddings[j])
            shared_embedings = (
                None
                if share_embedding_j == -1
                else self.prior_network[share_embedding_j].program_embedding
            )
            prior_neworkj = SimplePriorModel(
                series_length=series_length,  # 2x factor
                num_programs=num_programs[j],
                embedding_dim=embedding_dim,
                typ=prior_type,
                provided_emb=shared_embedings,
            )
            self.add_module("prior_network_layer" + str(j), prior_neworkj)  ## ** NEW
            self.prior_network.append(
                prior_neworkj
            )  ## does this get added to module params ?
            # self.add_module
        self.log_softmax = nn.LogSoftmax(dim=1)
        print("[JoinedPriorModel] list(model.parameters()) : ", list(self.parameters()))
        print(
            "[JoinedPriorModel] list(model.named_parameters()) : ",
            list(self.named_parameters()),
        )
        ##  --- check if prior nwtworks are actually part of params

    @overrides
    def forward(
        self,
        series: torch.Tensor,
        label_text=None,
        program_selection_method: str = None,
    ) -> Tuple[List[torch.IntTensor], torch.FloatTensor, torch.FloatTensor]:
        # -> torch.Tensor: # -> Dict[str,torch.Tensor]:
        assert program_selection_method in ["sample", "factored_argmax"]
        sampled_program: List[torch.IntTensor] = []
        sampled_program_logprob: torch.FloatTensor = 0.0
        sampled_program_emb_list = []
        for i in range(self.num_layers):
            program_dist_vals = self.prior_network[i].forward(
                series
            )  # bs, num_programs
            program_dist_logits_i = program_dist_vals["logits"]
            if program_selection_method == "sample":
                (
                    sampled_programi,
                    sampled_program_embi,
                    sampled_program_logprobi,
                ) = self.prior_network[i].sample(logits=program_dist_logits_i)
            else:  # argmax
                (
                    sampled_programi,
                    sampled_program_embi,
                    sampled_program_logprobi,
                ) = self.prior_network[i].get_argmax(logits=program_dist_logits_i)
            sampled_program.append(sampled_programi)
            sampled_program_logprob += sampled_program_logprobi
            sampled_program_emb_list.append(sampled_program_embi)
        sampled_program_emb: torch.FloatTensor = torch.cat(
            sampled_program_emb_list, dim=1
        )  # bs,e1+e2+e3
        # print("[JoinedPriorModel] series = ", series.size())
        # print("[JoinedPriorModel] sampled_program = ", sampled_program)
        # sampled_program_i, sampled_program_j, sampled_program_k = sampled_program[0], sampled_program[1], \
        #                                                           sampled_program[2]
        return sampled_program, sampled_program_emb, sampled_program_logprob

    def forward_vertical(
        self, series: torch.Tensor
    ) -> Tuple[torch.IntTensor, torch.FloatTensor, torch.FloatTensor]:
        # print("[forward_vertical] : self.vertical_prior_network = ", self.vertical_prior_network.__str__ )
        # print("[forward_vertical] : series = ", series.size())
        program_dist_vals = self.vertical_prior_network.forward(series)
        program_dist_logits_i = program_dist_vals["logits"]
        (
            sampled_program,
            sampled_program_emb,
            sampled_program_logprob,
        ) = self.vertical_prior_network.sample(logits=program_dist_logits_i)
        return sampled_program, sampled_program_emb, sampled_program_logprob

    def get_prob_of_sample(
        self, series: torch.FloatTensor, sampled_program: List[torch.IntTensor]
    ) -> torch.FloatTensor:
        # TODO: is this for a single datum or for a batch ?
        sampled_program_logprob: torch.FloatTensor = 0.0
        for i in range(self.num_layers):
            program_dist_vals = self.prior_network[i].forward(
                series
            )  # bs, num_programs ?
            program_dist_logits_i = program_dist_vals["logits"]
            sampled_programi = sampled_program[i]
            sampled_program_logprobi = self.prior_network[i].get_prob(
                logits=program_dist_logits_i, action=sampled_programi
            )
            sampled_program_logprob += sampled_program_logprobi
        return sampled_program_logprob

    def get_argmax(self, dist_values: List[torch.Tensor]):
        """
        :param dist_values:
        :return:
        As of now doing argmax independently for each layer
        """
        ret = []
        raise NotImplementedError

    def __getitem__(self, item):
        return self.prior_network[item]


class LabelTextModel(nn.Module):
    def __init__(self, typ: str = None, vocab_size: int = None, embedding_dim: int = 5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.typ = typ
        if typ == "lstm":
            self.model = None
            self.label_output_dim: int = None
        elif typ == "avg_emb":
            self.program_embedding = nn.Embedding(vocab_size, embedding_dim)
            self.model = None
            self.label_output_dim: int = embedding_dim
        else:
            assert False, "Not supported tpye for LabelTextModel"

    @overrides
    def forward(
        self, label_text: torch.LongTensor
    ) -> Tuple[torch.Tensor, Dict[Any, Any]]:
        # return an encoding of the text
        ret = None, {}
        if self.typ == "lstm":
            raise NotImplementedError
        elif self.typ == "avg_emb":
            # label_text_emb : bs, label_size
            # print("label_text: ", label_text.size())
            if (
                len(label_text.size()) == 1
            ):  # TODO - when using label field instead of text field
                label_text = label_text.unsqueeze(1)
            label_text_emb = self.program_embedding(
                label_text
            )  # bs, label_size, emb_Size
            avg_label_emb = torch.mean(label_text_emb, dim=1)  # bs, emb_Size
            ret = avg_label_emb, {}
        else:
            assert False, "Not supported tpye for LabelTextModel"
        return ret


class JoinedPosteriorModel(nn.Module):
    def __init__(
        self,
        series_length: int = None,
        num_layers: int = 3,
        num_programs: List[int] = None,
        prior_type: str = "simple_posterior",  # simple_prior,simple_posterior
        share_embeddings: str = "-1_-1_-1",
        embedding_dim: int = 5,
        label_embedding_dim: int = 5,
        label_model_typ: str = "avg_emb",
        debug_mode: bool = False,  # simulate prior model by ignoring label
        vocab_size: int = 4,  # TODO-vocabsize
        use_vertical_opers: bool = False,
    ):

        super().__init__()
        self.prior_network: List[SimplePriorModel] = []
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.use_vertical_opers = use_vertical_opers
        self.prior_type = prior_type
        self.share_embeddings = share_embeddings
        if use_vertical_opers:
            raise NotImplementedError
        self.debug_mode = debug_mode
        assert not debug_mode

        share_embeddings = share_embeddings.split("_")

        if self.debug_mode:
            label_output_dim = 0
            print("label_output_dim = ", label_output_dim)
        else:
            self.label_model = LabelTextModel(
                embedding_dim=label_embedding_dim,
                typ=label_model_typ,
                vocab_size=vocab_size,
            )
            label_output_dim = self.label_model.label_output_dim

        for j in range(num_layers):
            assert prior_type in ["simple_posterior"]
            prior_neworkj_type = prior_type
            if self.debug_mode:
                prior_neworkj_type = "simpleprior"
            share_embedding_j = int(share_embeddings[j])
            shared_embedings = (
                None
                if share_embedding_j == -1
                else self.prior_network[share_embedding_j].program_embedding
            )
            prior_neworkj = SimplePriorModel(
                series_length=series_length,
                num_programs=num_programs[j],
                embedding_dim=embedding_dim,
                typ=prior_neworkj_type,
                posterior_addition_dim=label_output_dim,
                provided_emb=shared_embedings,
            )
            self.add_module(
                "posterior_network_layer" + str(j), prior_neworkj
            )  ## ** NEW
            self.prior_network.append(prior_neworkj)
        self.log_softmax = nn.LogSoftmax(dim=1)
        # print("[JoinedPosteriorModel] list(model.parameters()) : ", list(self.parameters()))
        print(
            "[JoinedPosteriorModel] list(model.named_parameters()) : ",
            list(self.named_parameters()),
        )

    @overrides
    def forward(
        self,
        series: torch.Tensor,
        label_text: torch.LongTensor,
        program_selection_method: str = None,
    ) -> Tuple[List[torch.IntTensor], torch.Tensor, torch.FloatTensor]:

        # TODO - changes for Gumbel Softmax Reparametrization option
        print(
            "forward: [JoinedPosteriorModel] list(model.named_parameters()) : ",
            list(self.named_parameters()),
        )
        sampled_program: List[torch.IntTensor] = []
        sampled_program_logprob: torch.FloatTensor = 0.0
        sampled_program_emb = []
        if self.debug_mode:
            text_emb = None
        else:
            text_emb, _ = self.label_model.forward(
                label_text
            )  # *** Should also provide series to this  ? Probably yes. TODO
            print("** text_emb: ", text_emb.size())

        for i in range(self.num_layers):
            # print("** text_emb: ", text_emb.size(), " || ")
            program_dist_vals = self.prior_network[i].forward(
                series, posterior_addition_feats=text_emb
            )  # bs, num_programs
            program_dist_logits_i = program_dist_vals["logits"]
            if program_selection_method == "sample":
                (
                    sampled_programi,
                    sampled_program_embi,
                    sampled_program_logprobi,
                ) = self.prior_network[i].sample(logits=program_dist_logits_i)
            else:
                raise NotImplementedError
            sampled_program.append(sampled_programi)
            sampled_program_logprob += sampled_program_logprobi
            sampled_program_emb.append(sampled_program_embi)
            print(
                "self.prior_network[i] = ",
                list(self.prior_network[i].named_parameters()),
            )
            assert self.prior_network[i] == self.__getattr__(
                "posterior_network_layer" + str(i)
            )

        # assert False
        sampled_program_emb = torch.cat(sampled_program_emb, dim=1)  # bs,e1+e2+e3
        # print("[JoinedPriorModel] series = ", series.size())
        # print("[JoinedPriorModel] sampled_program = ", sampled_program)
        # sampled_program_i, sampled_program_j, sampled_program_k = sampled_program[0], sampled_program[1], \
        #                                                           sampled_program[2]
        return sampled_program, sampled_program_emb, sampled_program_logprob

    def __getitem__(self, item):
        return self.prior_network[item]
