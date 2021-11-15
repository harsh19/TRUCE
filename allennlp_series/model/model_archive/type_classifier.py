from typing import Dict, List, Any
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn.initializers import InitializerApplicator
from allennlp_series.training.metrics.confusion_matrix import (
    ConfusionMatrix,
    F1MeasureCustom,
)
from allennlp.training.metrics.average import Average
from overrides import overrides
import torch
import torch.nn as nn

# from src.utils.allenai.tensor_utils import get_text_field_mask
from allennlp_series.model.model_archive.assembled_models import (
    ThreeLayerProgram,
    TwoLayerProgram,
)
from allennlp_series.model.model_archive.layout_models import (
    ThreeLayerLayoutPredictionProgram,
)
from allennlp_series.model.model_archive.old_operation_featurizer import (
    OperationFeaturizer,
    OperationFeaturizerTwoLayer,
    OperationFeaturizerTwoLayerChoice,
)
from allennlp_series.model.model_archive.operations import VertTwoSeriesConvOperator
import numpy as np
from collections import Counter
from allennlp_series.common.constants import *


@Model.register("type_clf")
class TypeClassifier(Model):
    """
    An AllenNLP Model that runs pretrained BERT,
    takes the pooled output, and adds a Linear layer on top.
    If you want an easy way to use BERT for classification, this is it.
    Note that this is a somewhat non-AllenNLP-ish model architecture,
    in that it essentially requires you to use the "bert-pretrained"
    token indexer, rather than configuring whatever indexing scheme you like.

    See `allennlp/tests/fixtures/bert/bert_for_classification.jsonnet`
    for an example of what your config might look like.

    Parameters
    ----------
    vocab : ``Vocabulary``
    bert_model : ``Union[str, BertModel]``
        The BERT model to be wrapped. If a string is provided, we will call
        ``BertModel.from_pretrained(bert_model)`` and use the result.
    num_labels : ``int``, optional (default: None)
    num_labels : ``int``, optional (default: None)
        How many output classes to predict. If not provided, we'll use the
        vocab_size for the ``label_namespace``.
    index : ``str``, optional (default: "bert")
        The index of the token indexer that generates the BERT indices.
    label_namespace : ``str``, optional (default : "labels")
        Used to determine the number of classes if ``num_labels`` is not supplied.
    trainable : ``bool``, optional (default : True)
        If True, the weights of the pretrained BERT model will be updated during training.
        Otherwise, they will be frozen and only the final linear layer will be trained.
    use_ig: ``bool``, optional(default: False)
        If True, will incorporate IG features
    initializer : ``InitializerApplicator``, optional
        If provided, will be used to initialize the final linear layer *only*.
    latent clusters
        (2) Mark the closest cluster. update the marked cluster mean using moving average. add a loss term to data representation to come closer to the mean
        When predicting negative, move away from all negative chains ?
        (1) May be first visualize CLS pooled representation on which the classifier operates
    """

    def __init__(
        self,
        vocab: Vocabulary,
        dropout: float = 0.0,
        num_labels: int = 5,
        index: str = "bert",
        feature_size: int = 4,
        seq_length: int = 12,
        label_namespace: str = "labels",
        clf_type: str = "linear",
        series_type: str = SERIES_TYPE_SINGLE,
        operations_set_type: str = None,
        l1_loss_wt: float = 0.0,
        num_cols_in_multiseries: int = 3,
        operations_model_type: str = None,
        negative_class_wt: float = 1.0,
        run_without_norm: bool = False,
        initializer: InitializerApplicator = InitializerApplicator(),
    ) -> None:
        super().__init__(vocab)
        num_features: int = feature_size * seq_length
        in_features = num_features
        out_features = num_labels
        self._dropout = torch.nn.Dropout(p=dropout)
        self.series_type = series_type
        self.num_cols_in_multiseries = num_cols_in_multiseries
        self.run_without_norm = run_without_norm
        if run_without_norm:
            raise NotImplementedError

        assert clf_type in ["linear", "two_layer", "rnn"]
        self._clf_type = clf_type
        if series_type == SERIES_TYPE_MULTI:
            # print(num_cols_in_multiseries, in_features)
            in_features *= num_cols_in_multiseries
            # print(num_cols_in_multiseries, in_features)
            # assert False
        if clf_type == "linear":
            h = in_features
            layer2 = torch.nn.Linear(h, out_features)
            self._classification_layer = nn.Sequential(layer2)
        elif clf_type == "two_layer":
            h = int(in_features / 4)
            layer1 = torch.nn.Linear(in_features, h)
            layer2 = torch.nn.Linear(h, out_features)
            self._classification_layer = nn.Sequential(layer1, nn.ReLU(), layer2)
        elif clf_type == "rnn":
            self._lstm_h = h = 16
            self._lstm_layers = 2
            self._lstm = nn.LSTM(
                feature_size, h, self._lstm_layers, batch_first=True
            )  # provided batch, seq, feature
            layer2 = torch.nn.Linear(h, out_features)
            self._classification_layer = nn.Sequential(nn.ReLU(), layer2)

        self._accuracy = ConfusionMatrix(num_labels=num_labels)  # CategoricalAccuracy()
        self._f1 = F1MeasureCustom()
        # self._loss = torch.nn.CrossEntropyLoss()
        if negative_class_wt is None:
            self._loss = torch.nn.CrossEntropyLoss()
        else:
            self._loss = torch.nn.CrossEntropyLoss(
                weight=torch.tensor([negative_class_wt, 1.0])
            )
        self._index = index
        print("**** ========>>>>>>>>>>>>>>> in_features = ", in_features)
        initializer(self._classification_layer)

    def forward(
        self,  # type: ignore
        feats=None,
        label: torch.LongTensor = None,
        col=None,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor]
            From a ``TextField`` (that has a bert-pretrained token indexer)
        ig_tokens: Dict[str, torch.LongTensor]
            From a ``TextField`` (linearized Influence Graph)
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``
        metadata: Metadata from the dataset, optional
            From a ``MetaDataField``
        Returns
        -------
        An output dictionary consisting of:

        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            unnormalized log probabilities of the label.
        probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            probabilities of the label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """

        # pooled = self._dropout(pooled)

        # apply classification layer
        if self._clf_type == "rnn":
            bs = feats.size()[0]
            # print("bs = ", bs)
            print(feats.size())
            feats = torch.transpose(feats, 1, 2)
            print(feats.size())
            h0 = torch.zeros(self._lstm_layers, bs, self._lstm_h)
            c0 = torch.zeros(self._lstm_layers, bs, self._lstm_h)
            # output, (hn, cn) = self._rnn(input, (h0, c0))
            feats_new, (hn, cn) = self._lstm(feats, (h0, c0))
            last_output = feats_new[:, -1, :]
            print(last_output.size())
            last_output = last_output.view(bs, -1)
            logits = self._classification_layer(last_output)
        else:
            bs = feats.size()[0]
            # print("bs = ", bs)
            # print("feats => ", feats.size())
            feats = feats.view(bs, -1)
            logits = self._classification_layer(feats)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        output_dict = {"logits": logits, "probs": probs}
        output_dict["cols"] = [metadata_i["col"] for metadata_i in metadata]

        # print("logits = ", logits)
        # print("label = ", label)

        if label is not None:
            self._accuracy(logits, label)
            probs = torch.nn.functional.softmax(logits, dim=1)
            self._f1(logits, label, probs=probs)
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            output_dict["gold_label"] = label.data.cpu().numpy()
            output_dict["id"] = [metadata_i["id"] for metadata_i in metadata]
            print(col[0], label[0])

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if reset:
            print("CM = ", self._accuracy)
        metrics = self._accuracy.get_metric(
            reset
        )  # {'accuracy': self._accuracy.get_metric(reset)}
        metrics.update(self._f1.get_metric(reset))
        return metrics


from allennlp_series.common.constants import *


@Model.register("type_program_clf")
class SimpleProgramModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        dropout: float = 0.0,
        num_labels: int = 5,
        index: str = "bert",
        seq_length: int = 12,
        label_namespace: str = "labels",
        clf_type: str = "linear",
        operations_model_type: str = "operations",  # operations, operations_twolayer
        operations_set_type: str = "all",
        vert_operations_set_type: str = "vertconfiga",
        operations_use_signum: bool = False,
        operations_use_class_wise_l1: bool = False,
        operation_choice_num: int = None,
        operation_choice_num2: int = None,
        operations_acc_type: str = "max",
        l1_loss_wt: float = 0.1,
        kl_loss_wt: float = 1.0,
        prior_loss_wt: float = 1.0,
        negative_class_wt: float = None,
        series_type: str = SERIES_TYPE_SINGLE,
        num_cols_in_multiseries: int = 3,
        use_vertical_opers: bool = False,
        fname_to_dump: str = "preds.tsv",
        prior_model_type: str = "prior",
        posterior_model_type: str = "simple_posterior",
        use_inference_network: bool = False,
        use_inference_network_debug: bool = False,
        run_without_norm: bool = False,
        use_test_time_argmax: bool = False,
        initializer: InitializerApplicator = InitializerApplicator(),
    ):

        super().__init__(vocab)

        # print( " ****** vocab: self._index_to_token ", vocab._index_to_token.keys() )

        ## new
        # i2v_labels = vocab.get_index_to_token_vocabulary(label_namespace)
        # num_labels = len(i2v_labels.keys())
        # self.i2v_labels = i2v_labels
        # print("i2v_labels = ", i2v_labels)

        self.fname_to_dump = fname_to_dump
        self.use_vertical_opers = use_vertical_opers
        self.series_type = series_type
        self.prior_model_type = prior_model_type
        self.posterior_model_type = posterior_model_type
        self.num_cols_in_multiseries = num_cols_in_multiseries
        self.use_inference_network = use_inference_network
        self.run_without_norm = run_without_norm
        self.use_inference_network_debug = use_inference_network_debug
        if use_inference_network_debug:
            assert use_inference_network

        if use_inference_network:
            assert prior_model_type == "simpleprior"

        #### --------- Featur Extractors / Programs
        if operations_model_type == "operations_twolayer":
            self._featurizer: OperationFeaturizer = OperationFeaturizerTwoLayer(
                operations_set_type=operations_set_type,
                use_signum=operations_use_signum,
                use_class_wise_l1=operations_use_class_wise_l1,
                num_classes=num_labels,
            )
        elif operations_model_type == "operations":
            self._featurizer: OperationFeaturizer = OperationFeaturizer(
                operations_set_type=operations_set_type,
                use_signum=operations_use_signum,
            )
        elif operations_model_type == "operations_twolayer_choice":
            self._featurizer: OperationFeaturizer = OperationFeaturizerTwoLayerChoice(
                operations_set_type=operations_set_type,
                use_signum=operations_use_signum,
                choice_num=operation_choice_num,
                choice_num2=operation_choice_num2,
                num_classes=num_labels,
            )

        elif operations_model_type == "operations_threelayer_accum":
            self._featurizer: nn.Module = ThreeLayerProgram(
                operations_conv_type=operations_set_type,
                operations_acc_type=operations_acc_type,
                run_without_norm=run_without_norm,
                series_type=series_type,
            )

        elif operations_model_type == "operations_twolayer_accum":
            self._featurizer: nn.Module = TwoLayerProgram(
                operations_conv_type=operations_set_type,
                operations_acc_type=operations_acc_type,
                run_without_norm=run_without_norm,
                series_type=series_type,
            )

        elif operations_model_type == "operations_layout_pred_threelayer":
            self._featurizer: nn.Module = ThreeLayerLayoutPredictionProgram(
                operations_conv_type=operations_set_type,
                operations_acc_type=operations_acc_type,
                prior_model_type=prior_model_type,
                series_type=series_type,
                run_without_norm=run_without_norm,
                use_inference_network=use_inference_network,
                use_inference_network_debug=use_inference_network_debug,
                use_vertical_opers=use_vertical_opers,
                num_cols_in_multiseries=num_cols_in_multiseries,
                use_test_time_argmax=use_test_time_argmax,
                vert_operations_set_type=vert_operations_set_type,
            )

        if use_vertical_opers:
            assert series_type == SERIES_TYPE_MULTI
            if operations_model_type in ["operations_layout_pred_threelayer"]:
                self._vert_featurizer_out_channels = (
                    self._featurizer._vert_featurizer.out_channels
                )
                pass
                # ---- this is handled internally in ThreeLayerLayoutPredictionProgram
                # self._vert_featurizer: VertTwoSeriesConvOperatorChoice = VertTwoSeriesConvOperatorChoice(
                # inp_length=num_cols_in_multiseries,
                #     operations_type=vert_operations_set_type)
            else:
                self._vert_featurizer: VertTwoSeriesConvOperator = (
                    VertTwoSeriesConvOperator(
                        inp_length=num_cols_in_multiseries,
                        operations_type=vert_operations_set_type,
                    )
                )
                self._vert_featurizer_out_channels = self._vert_featurizer.out_channels

        self.operations_model_type = operations_model_type

        ######-------Classifier
        # define self.selector
        if operations_model_type in [
            "operations_threelayer_accum",
            "operations_twolayer_accum",
            "operations_layout_pred_threelayer",
        ]:
            num_features = self._featurizer.num_features
        else:
            num_features: int = (
                self._featurizer.out_channels * self._featurizer.num_features
            )  # feature_size * seq_length
        in_features = num_features
        if series_type == SERIES_TYPE_MULTI:
            if use_vertical_opers:
                in_features *= (
                    num_cols_in_multiseries + self._vert_featurizer_out_channels
                )
                if operations_model_type in ["operations_layout_pred_threelayer"]:
                    in_features += self._featurizer.prior_network.embedding_dim
            else:
                in_features *= num_cols_in_multiseries
        print("in_features = ", in_features)
        out_features = num_labels
        self.num_labels = num_labels
        self._dropout = torch.nn.Dropout(p=dropout)

        assert clf_type in ["linear", "two_layer", "three_layer"]
        self._clf_type = clf_type
        if operations_model_type in [
            "operations_twolayer",
            "operations_twolayer_choice",
        ]:
            if clf_type == "linear":
                # self._classification_layer = nn.Parameter(0.3*torch.ones(num_labels,
                #            self._featurizer.out_channels * self._featurizer.out_channels * self._featurizer.num_features))
                self._classification_layer = nn.Parameter(
                    0.3
                    * torch.randn(
                        (
                            num_labels,
                            self._featurizer.out_channels
                            * self._featurizer.out_channels
                            * self._featurizer.num_features,
                        )
                    )
                )
        else:
            if clf_type == "linear":
                h = in_features
                layer2 = torch.nn.Linear(h, out_features)
                self._classification_layer = nn.Sequential(layer2)
            elif clf_type == "two_layer":
                # h = int(in_features // 4)
                h = int(in_features // 2)
                layer1 = torch.nn.Linear(in_features, h)
                layer2 = torch.nn.Linear(h, out_features)
                self._classification_layer = nn.Sequential(layer1, nn.Sigmoid(), layer2)
            elif clf_type == "three_layer":
                h = int(in_features / 2)
                h2 = int(in_features / 4)
                # print(h,h2,out_features)
                layer1 = torch.nn.Linear(in_features, h)
                layer2 = torch.nn.Linear(h, h2)
                layer3 = torch.nn.Linear(h2, out_features)
                self._classification_layer = nn.Sequential(
                    layer1, nn.Sigmoid(), layer2, nn.Sigmoid(), layer3
                )
            initializer(self._classification_layer)

        print("self._classification_layer = ", self._classification_layer)
        self._accuracy = ConfusionMatrix(num_labels=num_labels)  # CategoricalAccuracy()
        # self._txt1accuracy = ConfusionMatrix(num_labels=num_labels//2)  # CategoricalAccuracy() ## hard-coding 2
        self._f1 = F1MeasureCustom()
        if negative_class_wt is None:
            self._loss = torch.nn.CrossEntropyLoss()
        else:
            self._loss = torch.nn.CrossEntropyLoss(
                weight=torch.tensor([negative_class_wt, 1.0])
            )
        self._index = index
        self._l1_loss_tracker = Average()
        self._prior_loss_tracker = Average()
        self._kl_loss_tracker = Average()
        self._kl_loss_wt = kl_loss_wt
        self._l1_loss_wt = l1_loss_wt
        self._prior_loss_wt = 1.0
        print("**** in_features = ", in_features)
        print(
            "**** negative_class_wt = ",
            negative_class_wt,
            "|| self._loss = ",
            self._loss,
        )
        self._selected_programs_in_epoch = {}  # class -> list of program indices

    def _get_l1_loss(self):
        """
        - This is only for use with programModel
        :return:
        """
        assert self._clf_type in ["linear", "two_layer", "three_layer"]
        assert self.operations_model_type in [
            "operations_threelayer_accum",
            "operations_twolayer_accum",
        ]
        # L1 on first layer weights only
        first_layer = self._classification_layer[0]
        l1_loss = torch.sum(torch.abs(first_layer.weight))
        # print("first_layer.weight = ", first_layer.weight)
        return l1_loss

    def forward(
        self,  # type: ignore
        feats=None,
        label: torch.LongTensor = None,
        col=None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:

        # # compute all features
        # computed_features = []
        # for operation in self.operations:
        #     computed_features.append(operation(inps))
        # computed_features = torch.stack(computed_features) # operations, batch, feature
        # computed_features = computed_features * self.selector
        # # predict
        # # get prediction loss
        # # get selector loss

        ## label_idx_batch = [metadata_i['label'] for metadata_i in metadata]
        ## word_to_labelidx_mapper = metadata[0].get('word_to_labelidx_mapper',None)
        # in case of text1 setting, this actss as combined label
        #   -- different string labels map to corresponding class integer label
        # the indices here are the original indices in dumped data pickle file
        # the 'label' has word indices. the predictions correspond to words
        # still need a word to label_idx mapper to convert the predicted class.: word_to_labelidx_mapper
        # text2 setting has different model code
        # print("label_idx_batch  = ", label_idx_batch )

        if self.use_vertical_opers:
            if self.operations_model_type in ["operations_layout_pred_threelayer"]:
                pass  # 3 handled inside lahyout program
            else:
                # col: bs, num_series, length
                more_cols = self._vert_featurizer(
                    col
                )  # more_cols: bs, num_series_more, length
                col = torch.cat([col, more_cols], dim=1)  # bs, n1+n2, length

        if self.use_inference_network:
            computed_features = self._featurizer(
                col, label
            )  ##-- add label to this call -- needed for inference network
        else:
            computed_features = self._featurizer(col)
        computed_features_dct = computed_features
        bs = col.size()[0]
        if self.operations_model_type == "operations":
            # computed_features: bs, out-channels, out-length
            feats = computed_features.view(bs, -1)
            logits = self._classification_layer(feats)
        elif self.operations_model_type in [
            "operations_twolayer",
            "operations_twolayer_choice",
        ]:
            feats = computed_features.view(
                bs, self.num_labels, -1
            )  # bs, numclasses, 27*27*8
            logits = torch.sum(
                self._classification_layer.unsqueeze(0) * feats, dim=2
            )  # -> bs, num_labels
        elif self.operations_model_type in [
            "operations_threelayer_accum",
            "operations_twolayer_accum",
        ]:
            feats = computed_features.view(bs, -1)
            logits = self._classification_layer(feats)  # -> bs, num_labels
        elif self.operations_model_type in ["operations_layout_pred_threelayer"]:
            # {'computed_program_output': out3,
            #  'sampled_program_emb': sampled_program_emb,
            #  'log_prob': sampled_program_logprob,
            #  'action': sampled_program}
            computed_features_dct = computed_features
            computed_program_output = computed_features_dct[
                "computed_program_output"
            ]  # bs,f1
            sampled_program_emb = computed_features_dct["sampled_program_emb"]  # bs,f2
            # print("computed_program_output:", computed_program_output.size())
            # print("sampled_program_emb:", sampled_program_emb.size())
            computed_features = torch.cat(
                [computed_program_output, sampled_program_emb], dim=1
            )  # bs,f1+f2
            logits = self._classification_layer(computed_features)  # -> bs, num_labels

        if np.random.rand() < 0.01:
            print(
                "[SimpleProgramModel] [forward] computed_features[bs=0] = ",
                computed_features[0],
            )

        probs = torch.nn.functional.softmax(logits, dim=-1)
        output_dict = {"logits": logits, "probs": probs}
        output_dict["cols"] = [metadata_i["col"] for metadata_i in metadata]

        if label is not None:

            # As we move to text output space, we will instead add bleu, meteor, etc.
            # add an if else statement here to that effect

            # In case we want to do some vae specific training like lagging inference etc., we need to create a custom training class
            # it could within its train loop keep tracking of losses, and decide which updater to use - so it updates only certain network

            self._accuracy(logits, label)
            probs = torch.nn.functional.softmax(logits, dim=1)
            self._f1(logits, label, probs=probs)
            # print("logits: ", logits.size())
            # print("label: ", label.size())
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            output_dict["gold_label"] = label.data.cpu().numpy()
            output_dict["series_att"] = [
                metadata_i.get("series_att", None) for metadata_i in metadata
            ]
            # print("output_dict[series_att] = ", output_dict["series_att"])
            output_dict["id"] = [metadata_i["id"] for metadata_i in metadata]

            if np.random.rand() < 0.05:
                print("[SimpleProgramModel] [forward]", col[0], label[0])

            # In case of inference network, also track the kl loss -- computed_features_dct[kl_loss]
            # add kl loss to total loss -- weighed
            # in case of using gumbel softmax, swithc off the reinforce part
            if self.use_inference_network:
                kl_loss = computed_features_dct["kl_loss"].mean()
                output_dict["kl_loss"] = self._kl_loss_wt * kl_loss
                print("output_dict[kl_loss] = ", output_dict["kl_loss"])
                output_dict["loss"] += output_dict["kl_loss"]
                self._kl_loss_tracker(output_dict["kl_loss"].data.cpu().item())

            if self._l1_loss_wt > 0:
                if self.operations_model_type in [
                    "operations_threelayer_accum",
                    "operations_twolayer_accum",
                ]:
                    output_dict["l1_loss"] = self._l1_loss_wt * self._get_l1_loss()
                    output_dict["loss"] += output_dict["l1_loss"]
                    self._l1_loss_tracker(output_dict["l1_loss"].data.cpu().item())
                else:
                    output_dict["l1_loss"] = (
                        self._l1_loss_wt * self._featurizer.get_l1_loss()
                    )
                    output_dict["loss"] += output_dict["l1_loss"]
                    self._l1_loss_tracker(output_dict["l1_loss"].data.cpu().item())

            if self.operations_model_type == "operations_layout_pred_threelayer":

                # computed_features_dct
                log_prob_action = computed_features_dct["log_prob"]  # bs,1

                # print(" MODE: self.training = ", self._featurizer.training)
                # print(" **** log_prob_action = ", log_prob_action)
                # *****. gradient is E_z [  \delta_phi(log p_phi(z)) * p_theta(y|z) ]
                # i.e. logprobaction scaled by (detached) p_theta -- not log p_theta
                # reward = -loss.detach() # lower classifier loss is better reward ---OLD
                # NEW:
                # print("label.view(1,-1): ", label.view(1,-1).size(), label) # bs

                reward = probs.gather(
                    1, label.view(-1, 1)
                )  # get p_theta(y) values -- bs,1
                ## TODO ---->>  should be using logprobs instead of probs as rewards
                reward = (
                    reward.detach()
                )  # -- only prior should be updated through this term
                # TODO: in above reward, weighby class weight - same as what used to comute classifierloss

                prior_objective_to_maximize = torch.mean(reward * log_prob_action)
                prior_loss = -prior_objective_to_maximize
                output_dict["prior_loss"] = self._prior_loss_wt * prior_loss
                output_dict["loss"] += output_dict["prior_loss"]
                self._prior_loss_tracker(output_dict["prior_loss"].data.cpu().item())

                ###
                if self.prior_model_type == "prior":
                    # print("computed_features_dct['action'] = ", computed_features_dct['action'])
                    for i, action_b in enumerate(
                        computed_features_dct["action"].data.cpu().numpy()
                    ):
                        label_b = output_dict["gold_label"][i]
                        if label_b not in self._selected_programs_in_epoch:
                            self._selected_programs_in_epoch[label_b] = []
                        self._selected_programs_in_epoch[label_b].append(action_b)
                else:
                    # print("computed_features_dct['action'] = ", computed_features_dct['action'])
                    actions0 = computed_features_dct["action"][0].data.cpu().numpy()
                    actions1 = computed_features_dct["action"][1].data.cpu().numpy()
                    actions2 = computed_features_dct["action"][2].data.cpu().numpy()
                    if self.use_vertical_opers:
                        actions3 = computed_features_dct["action"][3].data.cpu().numpy()
                    for i, action_b in enumerate(actions0):
                        if self.series_type == SERIES_TYPE_MULTI:
                            if self.use_vertical_opers:
                                label_b = output_dict["gold_label"][
                                    i // (self.num_cols_in_multiseries + 1)
                                ]  # 0,1,2->0
                                # because of additional vertical operator
                            else:
                                label_b = output_dict["gold_label"][
                                    i // self.num_cols_in_multiseries
                                ]  # 0,1->0  2,3->1
                        else:
                            label_b = output_dict["gold_label"][i]
                        if label_b not in self._selected_programs_in_epoch:
                            self._selected_programs_in_epoch[label_b] = []
                        if self.use_vertical_opers:
                            action3_cur = actions3[
                                i // (self.num_cols_in_multiseries + 1)
                            ]  # only bs num of vert opers
                            # hor opers are bs*numseries in count
                            self._selected_programs_in_epoch[label_b].append(
                                [actions0[i], actions1[i], actions2[i], action3_cur]
                            )
                        else:
                            self._selected_programs_in_epoch[label_b].append(
                                [actions0[i], actions1[i], actions2[i]]
                            )

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if reset:
            print("[SimpleProgramModel][get_metrics] CM = ", self._accuracy)
        metrics = self._accuracy.get_metric(
            reset
        )  # {'accuracy': self._accuracy.get_metric(reset)}

        if self._l1_loss_wt > 0:
            l1_loss = self._l1_loss_tracker.get_metric(reset)
            metrics.update({"l1_loss": l1_loss})
            if reset:
                first_layer = self._classification_layer[0]
                print("**** [L1]: first_layer = ", first_layer.weight)

        if self.use_inference_network:
            kl_loss = self._kl_loss_tracker.get_metric(reset)
            metrics.update({"kl_loss": kl_loss})

        if self.operations_model_type == "operations_layout_pred_threelayer":
            prior_loss = self._prior_loss_tracker.get_metric(reset)
            metrics.update({"loss_prior": prior_loss})
            if reset:
                print(" ====>>>>> self._selected_programs_in_epoch :")
                for label, indices in self._selected_programs_in_epoch.items():
                    print(" --> self._selected_programs_in_epoch : LABEL = ", label)
                    if self.prior_model_type == "prior":
                        print(
                            " --> self._selected_programs_in_epoch :indices = ",
                            Counter(indices),
                        )
                    else:
                        print(
                            " --> self._selected_programs_in_epoch :indices = ",
                            Counter(
                                [
                                    "".join([str(val) for val in indicesi])
                                    for indicesi in indices
                                ]
                            ),
                        )
                        for j in range(len(indices[0])):
                            print(
                                " --> Layer = ",
                                j + 1,
                                " self._selected_programs_in_epoch :indices = ",
                                Counter([indicesi[j] for indicesi in indices]),
                            )
                        print(
                            " ---> self._featurizer.prior_network[i].layer1.weight as follows"
                        )
                        print(
                            " ---> self._featurizer.prior_network[0]: ",
                            torch.sum(self._featurizer.prior_network[0].layer1.weight),
                        )
                        print(
                            " ---> self._featurizer.prior_network[1] ",
                            torch.sum(self._featurizer.prior_network[1].layer1.weight),
                        )
                        print(
                            " ---> self._featurizer.prior_network[2] ",
                            torch.sum(self._featurizer.prior_network[2].layer1.weight),
                        )
                        if self.use_vertical_opers:
                            print(
                                " ---> self._featurizer.vertical_prior_network ",
                                torch.sum(
                                    self._featurizer.prior_network.vertical_prior_network.layer1.weight
                                ),
                            )
                self._selected_programs_in_epoch = {}

        metrics.update(self._f1.get_metric(reset))
        return metrics


from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
import json


@Predictor.register("type_program_clf_predictor")
class NewPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._model = model
        self._cnt = 0
        fname_to_dump = self._fname_to_dump = model.fname_to_dump
        fw = open(fname_to_dump, "w")
        fw.close()
        self._pred_label_dist = {1: 0, 0: 0}
        self._total_cnt = 0
        if torch.cuda.is_available():
            self._model.cuda()

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        self._total_cnt += 1
        fw = open(self._fname_to_dump, "a")
        vals = []
        for col in outputs["cols"]:
            vals.append(json.dumps(col))
        vals.extend(
            [
                str(outputs["probs"][0]),
                str(outputs["probs"][1]),
                str(outputs["gold_label"]),
            ]
        )
        fw.write("\t".join(vals))
        fw.write("\n")
        fw.close()
        return json.dumps(outputs) + "\n"
