import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from allennlp_series.common.constants import *
from overrides import overrides
from typing import Tuple, Dict, List, Any


class EnumerateAllPrior(nn.Module):
    # 1) conditional prior over programs.
    #     - support sampling; and getting entropy of the distribution
    # 2) handling program embedding.
    #     - support getting program embedding for a given program
    def __init__(
        self,
        series_length: int = None,
        embedding_dim: int = 5,
        num_programs: int = None,
        init_program_emb_with_onehot: bool = False,
        instances: Dict = None,
        programs=None,
        use_factorized_program_emb: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.series_length = series_length
        self.num_programs = num_programs
        self.init_program_emb_with_onehot = init_program_emb_with_onehot
        self.use_factorized_program_emb = use_factorized_program_emb
        self.programs = programs

        _weight = None
        if not use_factorized_program_emb:
            if init_program_emb_with_onehot:
                # assert self.num_programs == self.embedding_dim
                _weight = torch.tensor(
                    np.zeros((self.num_programs, self.embedding_dim), dtype=np.float32)
                )
                for i in range(self.num_programs):
                    _weight[i][i] += 1
            self.program_embeddings_joined = nn.Embedding(
                self.num_programs, self.embedding_dim, _weight=_weight
            )
            if torch.cuda.is_available():
                self.program_embeddings_joined = self.program_embeddings_joined.cuda()
        else:
            j = 0
            self.program_embeddings = {}
            self.typ_to_typ_mapper = {}
            # instances has 3 keys: locate, attend, combine
            # instances[key] is a list of instantiations of that type
            for typ, lst in instances.items():
                embedding_dim = self.embedding_dim // 3
                cnt = len(lst) # num of instances of this type
                typ_mapper = {oper.operator_name: i for i, oper in enumerate(lst)}
                if init_program_emb_with_onehot:
                    _weight = torch.tensor(
                        np.zeros((cnt, embedding_dim), dtype=np.float32)
                    )
                    for i in range(cnt):
                        _weight[i][i] += 1
                program_emb = nn.Embedding(cnt, embedding_dim, _weight=_weight)
                if torch.cuda.is_available():
                    program_emb = program_emb.cuda()
                self.add_module("program_emb" + str(j), program_emb)
                self.program_embeddings[typ] = getattr(self, "program_emb" + str(j))
                self.typ_to_typ_mapper[typ] = typ_mapper
                print("-- typ_mapper = ", typ_mapper)
                j += 1

        self.tmp = (1 / num_programs) * torch.ones(num_programs)  ##3 --- uniform init
        self.logits = nn.Parameter(self.tmp)  # logits

    def program_embedding(self, ii):
        if self.use_factorized_program_emb:
            # print("ii.data.cpu().item() = ", ii.data.cpu(), len(ii.size()))
            emb = []
            if len(ii.size()) == 0:
                ii = [ii]
            for i in ii:
                program = self.programs[i.data.cpu().item()]
                idx0 = torch.zeros_like(i)
                idx0.fill_(
                    self.typ_to_typ_mapper["locate"][program.locate.operator_name]
                )
                # print("idx = ", idx0)
                emb0 = self.program_embeddings["locate"](idx0)
                idx1 = torch.zeros_like(i)
                idx1.fill_(
                    self.typ_to_typ_mapper["attend"][program.attend.operator_name]
                )
                emb1 = self.program_embeddings["attend"](idx1)
                idx2 = torch.zeros_like(i)
                idx2.fill_(
                    self.typ_to_typ_mapper["combine"][program.combine.operator_name]
                )
                emb2 = self.program_embeddings["combine"](idx2)
                embi = torch.cat([emb0, emb1, emb2])
                emb.append(embi)
            emb = torch.stack(emb, dim=0)
            if emb.size()[0] == 1:
                emb = emb.view(-1)
            return emb
        else:
            return self.program_embeddings_joined(ii)

    @overrides
    def forward(self, series: torch.Tensor):
        bs = series.size()[0]
        logits = self.logits.unsqueeze(0).repeat(
            bs, 1
        )  # bs, length -> bs, num_programs
        # dist_programs = F.log_softmax(logits)  # bs, num_programs
        dist_programs = F.log_softmax(logits, dim=1)  # bs, num_programs
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

    def get_program_emb(self, action) -> torch.FloatTensor:
        return self.program_embedding(action)

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

    def entropy(self, probs=None, logprobs=None):
        # probs: BS,Z
        if (probs is None) == (logprobs is None):
            raise ValueError(
                "Either `probs` or `logprobs` must be specified, but not both."
            )
        if probs is None:
            probs = torch.exp(logprobs)
        dist: torch.distributions.Categorical = torch.distributions.Categorical(
            probs=probs
        )
        entropy = dist.entropy()
        return entropy.mean()

