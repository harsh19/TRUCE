from typing import Dict, List, Tuple, Union, Any
from collections import Counter
import numpy as np
import torch
from torch.nn.modules.rnn import LSTMCell
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from allennlp.modules.token_embedders import Embedding
from allennlp.training.metrics.entropy import Entropy

from allennlp_series.common.constants import *
from allennlp_series.training.metrics.confusion_matrix import (
    ConfusionMatrix,
    F1MeasureCustom,
)
from allennlp_series.common.util import utils



@Model.register("inference_model")
class InferenceModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        hidden_dim: int = 5,
        embedding_dim: int = 5,
        dropout: float = None,
        initializer: InitializerApplicator = None,
        num_programs: int = 6,
        inference_nw_label_type: str = "complete",
        arch_type="lstm",
        emb_pretrained_file: str = None,
        use_inferencenw_evals: bool = False
        # entropy_wt:float=0.0
    ) -> None:

        super().__init__(vocab)
        # super().__init__(vocab, **kwargs)

        self.arch_type = arch_type
        self.num_programs = num_programs
        self._target_embedding_dim = embedding_dim
        self._decoder_output_dim = hidden_dim
        # self.register_buffer("_last_average_loss", torch.zeros(1))
        self.use_inferencenw_evals = use_inferencenw_evals
        if self.use_inferencenw_evals:
            self.class_to_text = {z: [] for z in range(num_programs)}
        self.use_inferencenw_evals_outputs = {}
        num_words = self.vocab.get_vocab_size("tokens")
        if emb_pretrained_file is not None:
            self._target_embedder = Embedding(
                num_words,
                self._target_embedding_dim,
                pretrained_file=emb_pretrained_file,
            )
        else:
            self._target_embedder = Embedding(num_words, self._target_embedding_dim)
        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x
        # self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        # self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        self._decoder_input_dim = (
            self._target_embedding_dim
        )  # + self._target_embedding_dim
        assert arch_type in ["lstm", "bilstm", "bow"]
        if arch_type == "lstm":
            self._decoder_cell = torch.nn.LSTM(
                self._decoder_input_dim, self._decoder_output_dim, batch_first=True
            )
        elif arch_type == "bilstm":
            self._decoder_cell = torch.nn.LSTM(
                self._decoder_input_dim,
                self._decoder_output_dim,
                batch_first=True,
                bidirectional=True,
            )
        elif arch_type == "bow":
            self._decoder_output_dim = (
                self._target_embedding_dim
            )  # vocab_size = vocab.get_vocab_size('tokens')
        assert inference_nw_label_type in ['complete']
        self._output_projection_layer = Linear(
            self._decoder_output_dim, num_programs
        )
        self.inference_nw_label_type = inference_nw_label_type
        self._loss = torch.nn.CrossEntropyLoss()
        # self._loss = torch.nn.CrossEntropyLoss(size_average=False)
        self._accuracy = ConfusionMatrix(
            num_labels=num_programs
        )  # CategoricalAccuracy()
        self._f1 = F1MeasureCustom()
        self._entr = Entropy()
        # self.entropy_wt = entropy_wt
        print("*** _index_to_token ", vocab._index_to_token.keys())
        self.labels_are_there = True if "labels" in vocab._index_to_token else False
        ##
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        num_params = sum([np.prod(p.size()) for p in model_parameters])
        print("[INFERENCE NETWORK] num params inference network = ", num_params)
        ##
        if initializer is not None:
            initializer(self)



    def process_batch(self, tokens):

        tokens = tokens["tokens"]
        batch_size = tokens.size()[0]
        # tokens: bs, ln
        if self.arch_type == "bow":
            embedded_input = self._target_embedder.forward(tokens)  # bs,ln,hs
            decoder_hidden_last = self._output_projection_layer(
                embedded_input
            )  # bs,ln,num_programs
            output_logits = torch.max(decoder_hidden_last, dim=1)  # bs,num_programs
        else:
            embedded_input = self._target_embedder.forward(tokens)  # bs,ln,hs
            lstm_out, (ht, ct) = self._decoder_cell(embedded_input)  # ,
            decoder_hidden_last = ht[-1]  # bs,h
            output_logits = self._output_projection_layer(
                decoder_hidden_last
            )  # bs,num_programs
        ret = {"logits": output_logits, "decoder_hidden_last": decoder_hidden_last}
        return ret


    def forward(  # type: ignore
        self,
        tokens: [str, torch.LongTensor],
        label: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
        series=None
    ) -> Dict[str, torch.Tensor]:

        # batch_size = tokens.size()[0]
        vals_loss = self.process_batch(tokens)
        output_logits = vals_loss["logits"]  # bs,numprograms
        output_dict = {"logits": output_logits}
        class_probabilities = F.softmax(output_logits, dim=-1)
        output_dict["probs"] = class_probabilities
        output_dict["log_probs"] = F.log_softmax(output_logits, dim=-1)

        entropy = self.entropy(output_logits)
        output_dict["entropy"] = entropy
        self._entr(output_logits)

        output_dict["raw_text"] = [metai["raw_text"] for metai in metadata]
        if label is not None:
            # entropy = self.entropy(output_logits)
            loss = self._loss(output_logits, label)
            # loss = loss - self.entropy_wt * entropy # encourage higher entropy to avoid collapse
            output_dict["loss"] = loss
            output_dict["gtlabel"] = label
            output_dict["raw_text_used"] = [
                metai["raw_text_used"] for metai in metadata
            ]
            output_dict["prediction"] = torch.argmax(class_probabilities, dim=1)
            self._accuracy(output_logits, label)
            self._f1(output_logits, label, probs=class_probabilities)
        else:
            output_dict["prediction"] = torch.argmax(class_probabilities, dim=1)
            if self.use_inferencenw_evals:
                assert len(output_dict["prediction"].data.cpu()) == len(
                    output_dict["raw_text"]
                )
                for z, txt in zip(
                    output_dict["prediction"].data.cpu(), output_dict["raw_text"]
                ):
                    self.class_to_text[z.item()].append(txt)
        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        metrics = {}
        if reset:
            print("CM = ", self._accuracy)
        metrics.update(self._accuracy.get_metric(reset))
        if self.labels_are_there:
            metrics.update(self._f1.get_metric(reset))
        metrics["entropy"] = self._entr.get_metric(reset).data.cpu().item()
        if self.use_inferencenw_evals:
            self.use_inferencenw_evals_outputs = {}
            for z, textlist in self.class_to_text.items():
                print("------z = ", z)
                print("#count = ", len(textlist), " --- ", textlist[:3])
                # for txt in textlist:
                #     print("text = ", txt)
                all_tokens = []
                for s in textlist:
                    all_tokens.extend(s.strip().split())
                print( "[use_inferencenw_evals] typescount", len(set(sorted(all_tokens))) )
                print(
                    "[use_inferencenw_evals] len(set(sorted(all_tokens)))",
                    len(set(sorted(all_tokens))),
                )
                print(
                    "[use_inferencenw_evals] sorted(list(dict(Counter(all_tokens)).items()), key=lambda x: -x[1]) [:25]",
                    sorted(
                        list(dict(Counter(all_tokens)).items()), key=lambda x: -x[1]
                    )[:25],
                )
                self.use_inferencenw_evals_outputs[z] = sorted(
                    list(dict(Counter(all_tokens)).items()), key=lambda x: -x[1]
                )
        return metrics


    def compute_kl_distance(self, log_posterior, log_prior):
        # return utils.compute_kl_distance(log_posterior=log_posterior, log_prior=log_prior)
        posterior = torch.exp(log_posterior)  # bs,Z
        kl_distance = torch.sum(posterior * (log_posterior - log_prior), dim=1)  # bs
        return kl_distance


    def entropy(self, logits_z):
        dist: torch.distributions.Categorical = torch.distributions.Categorical(
            logits=logits_z
        )
        entropy = dist.entropy()
        return entropy.mean()



##############################################################################################################


from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
import json
from overrides import overrides


@Predictor.register("inference_predictor")
class InferencePredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._model = model
        self._cnt = 0
        self._pred_label_dist = {1: 0, 0: 0}
        print(
            "*** label id to label: _index_to_token [labels]",
            model.vocab._index_to_token["labels"],
        )
        self._total_cnt = 0
        self.i2token = model.vocab._index_to_token["labels"]
        if torch.cuda.is_available():
            self._model.cuda()

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        self._total_cnt += 1
        outputs["gtlabel"] = self.i2token[outputs["gtlabel"]]
        outputs["prediction"] = self.i2token[outputs["prediction"]]
        outputs["correct"] = True
        if outputs["gtlabel"] != outputs["prediction"]:
            print("***** model made error on this instance")
            outputs["correct"] = False
        return json.dumps(outputs) + "\n"

