import torch.nn as nn
import math

from allennlp.nn.initializers import InitializerApplicator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.training.metrics.average import Average

from allennlp_series.model.prior_over_programs import *
from allennlp_series.model.cond_lm_model import ConditionalLanguageModel
from allennlp_series.model.model_archive.new_modules import *
from allennlp_series.model.trainable_modules import TrainableCombineModule
from allennlp_series.model.trainable_modules import TrainableLocateModule
from allennlp_series.model.baselines.seq2seq_encoder import Seq2SeqEncoder
from allennlp_series.model.inference_nw import InferenceModel
import allennlp_series.model.utils as utils
from allennlp_series.model.trainable_modules import (
    TrainableAttendModule as TrainableOneLayerAttendModule,
)
# TrainableTwoLayerAttendModule=TrainableAttendModule
from allennlp_series.model.trainable_modules import TrainableTwoLayerAttendModule


import logging
logger = logging.getLogger(__name__)


# ******************** ********************


def create_all_instances(
    depth: int = 1,
    use_combine_new_defn: bool = False,
    program_set="type1",
    trainable_locate: bool = False,
    trainable_attend: bool = False,
    init_trainable_with_hardcoded_vals: bool = False,
    num_attend_modules: int = 4,
    num_locate_modules: int = 3,
    use_position_encoding: bool = False,
    TrainableAttend=None,
    use_log_mode=False,
    use_learnable_bias_attend=False,
    use_learnable_bias_combine=False,
):
    assert depth == 1
    instances = {}

    instances["attend"] = []
    init_attend = init_trainable_with_hardcoded_vals
    if trainable_attend:
        fixed_module = False
    else:
        fixed_module = True
        init_attend = True

    if not init_attend:
        assert not fixed_module
        for j in range(num_attend_modules):
            instances["attend"].append(
                TrainableAttend(
                    operator_name="a" + str(j),
                    operator_init_type="none",
                    fixed_module=fixed_module,
                    learnable_bias=use_learnable_bias_attend,
                )
            )
    else:
        assert False
        # instances["attend"].append(
        #     TrainableAttend(
        #         operator_name="a1",
        #         operator_init_type="increase" if init_attend else "none",
        #         fixed_module=fixed_module,
        #         learnable_bias=use_learnable_bias_attend,
        #     )
        # )
        # instances["attend"].append(
        #     TrainableAttend(
        #         operator_name="a2",
        #         operator_init_type="decrease" if init_attend else "none",
        #         fixed_module=fixed_module,
        #         learnable_bias=use_learnable_bias_attend,
        #     )
        # )
        # if program_set in ["type2"]:
        #     instances["attend"].append(
        #         TrainableAttend(
        #             operator_name="a3",
        #             operator_init_type="peak" if init_attend else "none",
        #             fixed_module=fixed_module,
        #             learnable_bias=use_learnable_bias_attend,
        #         )
        #     )
        #     instances["attend"].append(
        #         TrainableAttend(
        #             operator_name="a4",
        #             operator_init_type="trough" if init_attend else "none",
        #             fixed_module=fixed_module,
        #             learnable_bias=use_learnable_bias_attend,
        #         )
        #     )

    instances["locate"] = []
    sz = LOCATE_SZ #12  # + 2
    # if trainable_attend:
    #     # sz=9+2 #
    #     sz = 10 + 2  #
    init_locate = init_trainable_with_hardcoded_vals
    if trainable_locate:
        fixed_module = False
    else:
        assert False
        # fixed_module = True
        # init_locate = True
    if not init_locate:
        assert not fixed_module
        for j in range(num_locate_modules):
            instances["locate"].append(
                TrainableLocateModule(
                    operator_type=None,
                    operator_name="l" + str(j),
                    inp_length=sz,
                    fixed_module=fixed_module,
                    use_position_encoding=use_position_encoding
                )
            )
    else:
        assert False
        # instances["locate"].append(
        #     TrainableLocateModule(
        #         operator_type="begin" if init_locate else None,
        #         operator_name="l1",
        #         inp_length=sz,
        #         fixed_module=fixed_module,
        #         use_position_encoding=use_position_encoding,
        #     )
        # )
        # instances["locate"].append(
        #     TrainableLocateModule(
        #         operator_type="middle" if init_locate else None,
        #         operator_name="l2",
        #         inp_length=sz,
        #         fixed_module=fixed_module,
        #         use_position_encoding=use_position_encoding,
        #     )
        # )
        # instances["locate"].append(
        #     TrainableLocateModule(
        #         operator_type="end" if init_locate else None,
        #         operator_name="l3",
        #         inp_length=sz,
        #         fixed_module=fixed_module,
        #         use_position_encoding=use_position_encoding,
        #     )
        # )

    instances["combine"] = [
        TrainableCombineModule(
            operator_type="combine_exists",
            operator_name="combine_exists",
            use_new_defn=use_combine_new_defn,
            log_mode=use_log_mode,
            learnable_bias=use_learnable_bias_combine,
        )
    ]
    return instances


model_type_to_testtimedefault = {
    "marginalize": "marginalize",
    "marginalize_new": "marginalize_new",
    "prior_reinforce": "sample",
}


def get_text_field_mask(
    text_field_tensors, num_wrapping_dims: int = 0
) -> torch.LongTensor:
    # if "mask" in text_field_tensors:
    # print("text_field_tensors = ", text_field_tensors)
    return text_field_tensors["tokens"] != 0
    # return text_field_tensors["mask"]






@Model.register("truce_method")
class TRUCEMethod(Model):

    def __init__(
        self,
        vocab: Vocabulary,
        inp_length=12,
        operations_conv_type: str = "all",
        operations_acc_type: str = "max",
        use_inference_network_debug: bool = None,
        test_time_method_type: str = None,
        embedding_dim: int = 6,
        num_labels: int = 5,
        negative_class_wt: float = 1.0,
        prior_loss_wt: float = 1.0,
        clf_loss_wt: float = 1.0,
        use_inference_network: bool = False,
        model_type: str = "marginalize",
        reinforce_baseline: str = None,  # None, 'mean_std'
        reinforce_num_samples: int = 3,
        init_program_emb_with_onehot: bool = False,
        use_combine_new_defn: bool = False,
        use_prior_potential: bool = False,
        train_unary_prior: bool = None,
        program_set_type: str = "type1",
        program_trainable_locate: bool = False,
        program_trainable_attend: bool = False,
        text_hidden_dim: int = 5,
        text_embedding_dim: int = 5,
        task_setup_type: str = "classification",  # classification, text
        text_eval_sanity_check_mode: bool = False,
        init_trainable_with_hardcoded_vals: bool = False,
        seq2seq_model_type="1layer",
        use_position_encoding: bool = False,
        num_attend_modules: int = 2,
        num_locate_modules: int = 3,
        use_factorized_program_emb: bool = False,
        attend_module_type: str = "one_layer",
        decoding_method: str = "greedy",
        sampling_top_p: float = 0.9,
        sampling_top_k: int = None,
        program_selection_method: str = "argmax",
        program_selection_method_topk: int = 10,
        use_activation_evals: bool = False,
        use_bertscore_evals: bool = False,
        use_inferencenw_evals: bool = False,
        model_name: str = None,
        predictor_file_name: str = None,
        inference_network_model_path: str = None,
        inference_network_frozen: bool = False,
        inferencenw_entropy_wt: float = 0.0,
        inferencenw_batchavg_entropy_wt: float = 0.0,
        mutual_kl_distance_wt: float = 0.0,
        mutual_kl_distance_use_log: bool = False,
        prior_entropy_wt: float = 0.0,
        prior_batchavg_entropy_wt: float = 0.0,
        use_prior_entropy_regularize: bool = False,
        klterm_weight: float = 1.0,
        reinforce_loss_coef: float = 0.1,
        kl_threshold: float = None,
        kl_annealing_type: str = None,
        kl_sanity_check: bool = False,
        use_bow_decoder: bool = False,
        simplify_outputs: bool = False,
        use_all_scores_list_tensor_l2: bool = False,
        program_score_l2_wt: float = 0.0,
        use_log_mode: bool = False,
        use_learnable_bias: bool = False,
        seq2seq_use_multitask: bool = False,
        seq2seq_multitask_loss_wt: float = 1.0,
        seq2seqmulti_num_heuristic_labels: int = None,
        pseudo_label_loss_weight_config: str = "default",
        inference_nw_label_type: str = "complete",
        add_yx_connection: bool = False,
        initializer: InitializerApplicator = InitializerApplicator(),
    ):

        super().__init__(vocab)
        self.simplify_outputs = simplify_outputs
        self.use_inference_network = use_inference_network
        if test_time_method_type is None:
            test_time_method_type = model_type_to_testtimedefault[model_type]
        assert test_time_method_type in [
            "sample",
            "argmax",
            "marginalize",
            "marginalize_new",
        ]
        self.test_time_method_type = test_time_method_type
        self.num_labels = num_labels
        self.negative_class_wt = negative_class_wt
        self.model_type = model_type
        self.reinforce_baseline = reinforce_baseline
        self.task_setup_type = task_setup_type
        if negative_class_wt != 1.0 and num_labels > 2:  # not yet implemented
            raise NotImplementedError
        self.program_set_type = program_set_type
        assert program_set_type in ["type1", "type2"]
        self.program_selection_method = program_selection_method
        self.program_selection_method_topk = program_selection_method_topk
        # assert program_selection_method in []
        # if program_set_type == 'type2':
        #    assert num_attend_modules == 4
        assert attend_module_type in ["one_layer", "two_layer"]
        if task_setup_type in ['text']:
            self.instances = instances = create_all_instances(
                use_combine_new_defn=use_combine_new_defn,
                program_set=program_set_type,
                trainable_locate=program_trainable_locate,
                trainable_attend=program_trainable_attend,
                init_trainable_with_hardcoded_vals=init_trainable_with_hardcoded_vals,
                num_attend_modules=num_attend_modules,
                num_locate_modules=num_locate_modules,
                use_position_encoding=use_position_encoding,
                TrainableAttend=TrainableOneLayerAttendModule
                if attend_module_type == "one_layer"
                else TrainableTwoLayerAttendModule,
                use_log_mode=use_log_mode,
                use_learnable_bias_attend=use_learnable_bias,
                use_learnable_bias_combine=use_learnable_bias,
            )
        assert model_type in [
            "marginalize",
            "marginalize_new",
            "prior_reinforce",
            "inference_nw",
        ]
        self.clf_loss_wt = clf_loss_wt
        self.reinforce_num_samples = reinforce_num_samples
        self.use_prior_potential = use_prior_potential
        if use_prior_potential:
            assert model_type == "marginalize_new"
        if train_unary_prior is None:
            train_unary_prior = False
            if use_prior_potential:
                train_unary_prior = True
        self.train_unary_prior = train_unary_prior
        self.text_eval_sanity_check_mode = text_eval_sanity_check_mode
        if text_eval_sanity_check_mode:
            assert self.task_setup_type == "text"
        self.program_trainable_locate = program_trainable_locate
        self.program_trainable_attend = program_trainable_attend

        self.programs = []
        self.num_programs = 0
        dct = {}
        if task_setup_type in ['text']:
            for i in range(len(instances["locate"])):
                dct["locate"] = i
                for j in range(len(instances["attend"])):
                    dct["attend"] = j
                    for k in range(len(instances["combine"])):
                        dct["combine"] = k
                        programi = SimpleProgramType(dct, instances)
                        # add module
                        self.num_programs += 1
                        self.add_module("program" + str(self.num_programs), programi)
                        self.programs.append(programi)
                        logger.info(
                            f"[NewThreeLayerLayoutPredictionProgram]: {self.num_programs - 1} {programi.__str__}"
                        )

        self._add_yx_connection = add_yx_connection
        if self.task_setup_type not in ["seq2seq", "unconditional_lm"]:
            self.prior_model = EnumerateAllPrior(
                num_programs=self.num_programs,
                embedding_dim=embedding_dim,
                init_program_emb_with_onehot=init_program_emb_with_onehot,
                instances=instances,
                use_factorized_program_emb=use_factorized_program_emb,
                programs=self.programs,
            )
        else:
            self.prior_model = False

        program_emb_size = embedding_dim
        self.text_model = ConditionalLanguageModel(
            vocab=vocab,
            initializer=initializer,
            program_emb_size=program_emb_size,
            eval_sanity_check_mode=text_eval_sanity_check_mode,
            hidden_dim=text_hidden_dim,
            embedding_dim=text_embedding_dim,
            use_activation_evals=use_activation_evals,
            model_programs=self.programs,
            model_name=model_name,
            use_bertscore_evals=use_bertscore_evals,
            use_bow_decoder=use_bow_decoder,
            decoding_method=decoding_method,
            sampling_top_p=sampling_top_p,
            sampling_top_k=sampling_top_k,
        )

        self.use_inferencenw_evals = use_inferencenw_evals
        if self.use_inference_network:
            self.inference_network = InferenceModel(
                vocab=vocab,
                hidden_dim=text_hidden_dim,  # add separate param for inference n/w ?
                embedding_dim=text_hidden_dim,  # add separate param for inference n/w ?
                initializer=initializer,
                num_programs=self.num_programs,
                # num_programs2 = 2,
                inference_nw_label_type=inference_nw_label_type,
                use_inferencenw_evals=use_inferencenw_evals,
                arch_type="bilstm"  # ,
                # entropy_wt=inferencenw_entropy_wt
            )
            self._reward_tracker = Average()
            self._elbo_tracker = Average()
            # self._priorloss_tracker = Average()
            self.running_mean = None
            self.reinforce_loss_coef = reinforce_loss_coef  # 0.1

        self._kl_distance_tracker = Average()
        self._mutualkl_distance_tracker = Average()
        self._pseudo_label_loss_value_tracker = Average()
        self.inferencenw_entropy_wt = inferencenw_entropy_wt
        self.inferencenw_batchavg_entropy_wt = inferencenw_batchavg_entropy_wt
        self.inference_network_model_path = inference_network_model_path
        self.inference_network_frozen = inference_network_frozen
        self.prior_entropy_wt = prior_entropy_wt
        self.mutual_kl_distance_wt = mutual_kl_distance_wt
        self.mutual_kl_distance_use_log = mutual_kl_distance_use_log
        self.use_prior_entropy_regularize = use_prior_entropy_regularize
        self.kl_threshold = kl_threshold  # 0.1
        self.kl_annealing_type = kl_annealing_type
        self.seq2seq_multitask_loss_wt = seq2seq_multitask_loss_wt
        self.seq2seq_use_multitask = seq2seq_use_multitask
        self._multitask_heuristic_loss_tracker = Average()
        self.seq2seqmulti_num_heuristic_labels = seq2seqmulti_num_heuristic_labels

        if (
            self.task_setup_type in ["seq2seq", "classification_seq2seq"]
            or self._add_yx_connection
        ):
            self.seq2seq_encoder = Seq2SeqEncoder(
                vocab=vocab,
                initializer=initializer,
                model_type=seq2seq_model_type,
                embedding_dim=embedding_dim,
            )
            if self.seq2seq_use_multitask:
                self.seq2seq_multitask_predictor = nn.Linear(
                    embedding_dim, self.seq2seqmulti_num_heuristic_labels
                )

        ### text gen. auto evals
        # bleu, perplexity, meteor, rouge, cider
        self.use_bertscore_evals = use_bertscore_evals
        if use_bertscore_evals:
            # import bert_score
            # self.bert_scorer = bert_score.BERTScorer(lang="en", batch_size=3, rescale_with_baseline=True)
            # self._bert_p_r_f1_tracker = [Average()]
            pass

        ### others
        self._prior_loss_tracker = Average()
        self._clf_loss_tracker = Average()
        self._reward_tracker = Average()
        self._prior_entropy_tracker = Average()
        self._prior_batchavg_entropy_tracker = Average()
        self._posterior_batchavg_entropy_tracker = Average()
        self._prior_loss_wt = prior_loss_wt
        self._prior_batchavg_entropy_wt = prior_batchavg_entropy_wt
        self._z_chosen_counts = [0 for j in range(self.num_programs)]
        self._z_class_probs = [
            [Average() for k in range(self.num_labels)]
            for j in range(self.num_programs)
        ]
        self._z_class_wprob = [
            [Average() for k in range(self.num_labels)]
            for j in range(self.num_programs)
        ]

        ##
        self.model_name = model_name
        self.predictor_file_name = predictor_file_name
        print("****** vocab._index_to_token ---> ", vocab._index_to_token.keys())
        print("****** vocab[labels] ---> ", vocab._index_to_token.get("labels", None))

        if initializer is not None:
            initializer(self)

        if self.use_inference_network:
            print("self.inference_network = ", self.inference_network)
        self.klterm_weight = klterm_weight
        if self.kl_annealing_type is not None:
            if self.kl_annealing_type == "linear_0_005":
                self.klterm_weight = 0.0
            else:
                raise NotImplementedError

        if inference_network_model_path is not None:
            assert False
            # logger.info("===>>>>>> Loading generator from = %s", str(lm_weights_file))
            model_state = torch.load(inference_network_model_path)
            print("model_state: ", model_state.keys())
            self.inference_network.load_state_dict(model_state)
            # 0/0
        if inference_network_frozen:
            # assert False
            assert inference_network_model_path is not None
            self.inference_network.requires_grad = False
            for n, p in self.inference_network.named_parameters():
                p.requires_grad = False
            for n, p in self.inference_network.named_parameters():
                print(n, p.requires_grad)
            # 0/0

        # self._pseudo_label_loss = torch.nn.CrossEntropyLoss(ignore_index=0) # use label weights here
        # weights = [0.5, 1.0, 1.0, 1.0, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        # class_weights = torch.FloatTensor(weights).cuda()
        # self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        if pseudo_label_loss_weight_config == "default":
            self._pseudo_label_loss = torch.nn.CrossEntropyLoss(
                ignore_index=0
            )  # use label weights here
        else:
            weights = [1.0] * self.seq2seqmulti_num_heuristic_labels
            if pseudo_label_loss_weight_config == "weight_config1":
                heuristic_labels_list = {
                    0: 460,
                    18: 26,
                    12: 25,
                    19: 23,
                    20: 14,
                    14: 14,
                    21: 11,
                    6: 9,
                    15: 9,
                    13: 8,
                    27: 6,
                    7: 6,
                    8: 5,
                    24: 5,
                    9: 4,
                    25: 4,
                    26: 1,
                }
                # for c,cnt in heuristic_labels_list.items():
                for c in range(self.seq2seqmulti_num_heuristic_labels):
                    if c in heuristic_labels_list:
                        cnt = heuristic_labels_list[c]
                        weights[c] = 10.0 * 1.0 / cnt
            elif pseudo_label_loss_weight_config == "weight_config2":
                # todo
                heuristic_labels_list = {
                    0: 460,
                    18: 26,
                    12: 25,
                    19: 23,
                    20: 14,
                    14: 14,
                    21: 11,
                    6: 9,
                    15: 9,
                    13: 8,
                    27: 6,
                    7: 6,
                    8: 5,
                    24: 5,
                    9: 4,
                    25: 4,
                    26: 1,
                }
                weights = [1.0] * self.seq2seqmulti_num_heuristic_labels
                # for c, cnt in heuristic_labels_list.items():
                #     weights[c] = 1.0 / cnt
                for c in range(self.seq2seqmulti_num_heuristic_labels):
                    if c in heuristic_labels_list:
                        cnt = heuristic_labels_list[c]
                        weights[c] = 1.0 / cnt
            class_weights = torch.FloatTensor(weights).cuda()
            self._pseudo_label_loss = torch.nn.CrossEntropyLoss(
                ignore_index=0, weight=class_weights
            )  # use label weights here

        self.kl_sanity_check = kl_sanity_check
        self.use_all_scores_list_tensor_l2 = use_all_scores_list_tensor_l2
        self.program_score_l2_wt = program_score_l2_wt
        self.scores_l2_tracker = Average()

    def get_proby_givenz(
        self,
        z_embedding,
        label=None,
        label_text=None,
        metadata=None,
        mode="ppl",
        selected_program_id=None,
    ):

        if self.task_setup_type in ["text", "gt_program_text", "seq2seq"]:

            # print("[get_proby_givenz] z_embedding = ", z_embedding.size())
            vals = self.text_model.forward(
                z_embedding,
                target_tokens=label_text,
                metadata=metadata,
                mode=mode,
                selected_program_id=selected_program_id,
            )
            logprob_ylabel_given_z = vals["logprob_ylabel_given_z"]
            return logprob_ylabel_given_z, vals
            #### this return p(y=label|z)
            # as a sanity check, can provide ground truth z, and see what the decoder learns

        else:
            raise NotImplementedError



    def marginalize_new(
        self,
        series,
        label,
        label_text,
        predictor_network,
        metadata,
        use_prior_potential: bool = False,
        train_unary_prior: bool = True,
        heuristic_label=None
    ):

        bs = series.size()[0]
        if train_unary_prior:
            assert use_prior_potential

        vals = predictor_network(series)
        logits = vals["logits"]  # bs, program_space
        dist: torch.distributions.Categorical = torch.distributions.Categorical(
            logits=logits
        )

        all_logprobs_list = []
        all_scores_list = []

        approx_posterior_entropy = None
        if self.use_inference_network:
            approx_posterior_vals = self.inference_network.forward(
                tokens=label_text, label=label, series=series, metadata=metadata
            )
            approx_posterior_attached = approx_posterior_vals["log_probs"]  # bs, |Z|
            approx_posterior_batchavg_attached = torch.mean(
                approx_posterior_attached, dim=0
            )  # |Z|
            approx_posterior_detached = approx_posterior_attached.detach()
            approx_posterior_entropy = approx_posterior_vals["entropy"]
            # q(z|y,x)
            # detach to get approx_posterior - will be used for E_{qz}[log(p(y|z))-log(q(z)|p(z)))]
            # i.e. E_{qz}[log(p(y|z))]-KLlog(q(z)||p(z))]
            # separately, need to compute rewards for q(z)
            # which for q(z=i) is log(p(y|z=i))
            # i.e. want to maximize Ri*logq(z=i). Ri=log(p(y|z=i)
            # so add  -Ri*logq(z=i)  to the loss term for each z=i
            # make sure to used approx_posterior_attached
            #####
            # if factorized, it will have approx_posterior['locate'] and approx_posterior['attend']
            # then construct q(z=1) = q_attend(z=1_a) * q_locate(z=1_l) ?? need to think more here
            # iterate over programs. q(z=i) = q_attend(z=i_att) * q_locate(z=i_loc)
            # how to get i_loc and i_att -> from names ?
            ####

        rewards = []
        logprob_actions = []
        attendl2norms = []

        # 'locate': v1,
        # 'attend': v2,
        # 'combine_v3': v3,
        # 'combine_score': score,
        # 'ret': ret
        analysis_vals = {"locate": [], "attend": [], "attendpre": [], "combine_v3": []}

        if self._add_yx_connection:
            series_encoding = self.seq2seq_encoder.forward(series)

        for z in range(self.num_programs):

            action = torch.tensor(z)
            if torch.cuda.is_available():
                action = action.cuda()
            program = self.programs[z]
            # print("series = ", series)
            # print()
            # print("[marginalize] ==> z = ", z,
            #       " ||| label = ", label,
            #       " ||| label_text=", label_text)
            # _, score_z_x, attendl2norm = program.forward(series, get_score_also=True, l2_mode=True)
            program_vals = program.forward(
                series, get_score_also=True, l2_mode=True, analysis_mode=True
            )
            _, score_z_x, attendl2norm = program_vals["ret"]
            # score_z_x treated as score of log of score ?
            attendl2norms.append(attendl2norm)
            all_scores_list.append(score_z_x.unsqueeze(0))
            # print("[marginalize] ==> z = ", z, " ||  score_z_x = ", score_z_x,
            #       " ||| label = ", label,
            #       " ||| label_text=",label_text)

            z_embedding = predictor_network.get_program_emb(action=action)  # embsize
            z_embedding = z_embedding.unsqueeze(0).repeat(bs, 1)
            # print("="*33)
            # print("z_embedding = ", z_embedding)
            # print("label_text = ", label_text)
            # print("label = ", label)
            # print("="*33)
            # print("metadata = ", metadata)

            z_embedding_for_decoder = z_embedding
            if self._add_yx_connection:
                z_embedding_for_decoder = z_embedding_for_decoder + series_encoding
                # z_embedding -> bs, emb_size
                # series_encoding -> bs, emb_size

            logprob_y_given_z, _ = self.get_proby_givenz(
                z_embedding=z_embedding_for_decoder,
                label=label,
                label_text=label_text,
                metadata=metadata,
                selected_program_id=z,
            )
            # print("bs = ", bs)
            # print(logprob_y_given_z.size())
            assert logprob_y_given_z.size()[0] == bs
            # print("[marginalize] ==> z = ", z, " ||  logprob_y_given_z = ", logprob_y_given_z)
            logprob_y_given_z_numpy = logprob_y_given_z.data.cpu().numpy()
            if self.task_setup_type not in ["text", "seq2seq"]:
                for logprob_y_given_z_numpy_zidx in logprob_y_given_z_numpy:
                    for j, val in enumerate(logprob_y_given_z_numpy_zidx):
                        self._z_class_probs[z][j](val)

            # cur_prob = logprob_z.unsqueeze(1) + logprob_w_given_z.unsqueeze(1) + logprob_y_given_z # bs, num_labels
            unnormalized_logprobz_givenx = score_z_x  # bs
            if self.task_setup_type != "text":
                cur_prob_score = (
                    unnormalized_logprobz_givenx.unsqueeze(1) + logprob_y_given_z
                )  # bs, num_labels
            else:
                if self.use_inference_network:
                    # print("approx_posterior: ", approx_posterior.size())
                    # reconstruction term
                    # cur_prob_score = approx_posterior_detached[:,z] + logprob_y_given_z  # bs
                    # since we are training through marginalizing in this function ( as opoosed to using reinforce)
                    # simply add approx_posterior_attached[:,z] i.e. params of inference network are updated through this term
                    cur_prob_score = (
                        approx_posterior_attached[:, z] + logprob_y_given_z
                    )  # bs
                else:
                    cur_prob_score = (
                        unnormalized_logprobz_givenx + logprob_y_given_z
                    )  # bs

            if use_prior_potential:
                logprob_z = dist.log_prob(
                    action
                )  # bs   #.unsqueeze(0).repeat(bs,1) # bs
                # print("[marginalize] ==> z = ", z, "logprob_z = ", logprob_z, " ||| label = ", label)
                if not train_unary_prior:
                    logprob_z = logprob_z.detach()
                if self.task_setup_type != "text":
                    cur_prob_score += logprob_z.unsqueeze(1)  # bs, num_labels
                else:
                    cur_prob_score += logprob_z  # bs
                    if self.use_inference_network:
                        raise NotImplementedError

            # following is not needed here. this would be useful when using samples instead of marginalizing
            # if self.use_inference_network:
            #     reward_z = logprob_y_given_z
            #     rewards.append(reward_z)
            #     logprob_actions.append(approx_posterior_attached[:,z])
            #     # loss_qz = - ( reward_z * approx_posterior[z] )
            #     # need to maximinze Ri * log p(zi)

            # print("[marginalize] ==> z = ", z, " ||  cur_prob = ", cur_prob_score )
            all_logprobs_list.append(
                cur_prob_score.unsqueeze(0)
            )  # 1, bs, numlabels or 1, bs
            # all_logprobs_list.append(cur_prob.data.cpu().numpy())
            # print()

        # computing the normalizing factor
        # all_scores_list: numprograms,bs
        # normalizing : bs
        all_scores_list_tensor = torch.cat(all_scores_list, dim=0)  # nhum_programs, bs
        # p(z) = s(z)/Z
        # i.e. Z = \sum_z s(z) => log Z = logsumxp([log s(z1), log s(z2), .. ])
        all_scores_list_normalizer = torch.logsumexp(
            all_scores_list_tensor, dim=0
        )  # bs
        # p(z) = s(z)/Z i.e. log p(z) = log s(z) - log Z
        logprior_zgivenx = (
            all_scores_list_tensor - all_scores_list_normalizer.unsqueeze(0)
        )
        # nhum_programs, bs
        all_scores_list_normalizer = all_scores_list_normalizer.unsqueeze(1)  # bs,1

        attendl2norms = torch.stack(attendl2norms, dim=0)
        all_scores_list_tensor_l2 = torch.mean(attendl2norms)

        # very low prior entropy might be penalized. compute prior_entropy and add it to return dictionry
        # prior_entropy = None
        # if self.use_prior_entropy_regularize:
        prior_entropy = self.prior_model.entropy(logprior_zgivenx.t())
        self._prior_entropy_tracker(prior_entropy.mean().data.cpu().item())
        prior_batchavg = torch.mean(logprior_zgivenx.t(), dim=0).unsqueeze(
            0
        )  # bs,Z -> Z -> 1,Z
        prior_batchavg_entropy = self.prior_model.entropy(prior_batchavg)
        self._prior_batchavg_entropy_tracker(
            prior_batchavg_entropy.mean().data.cpu().item()
        )  #
        self.scores_l2_tracker(all_scores_list_tensor_l2.data.cpu().item())

        for z in range(self.num_programs):
            if self.task_setup_type != "text":
                all_logprobs_list[z] = (
                    all_logprobs_list[z] - all_scores_list_normalizer
                )  # bs,numn_labels
                # and bs,1 -> bs,num_labels
                assert not self.use_inference_network
            else:
                if self.use_inference_network:
                    pass
                    # approx posterior through inference network is already normalized
                    # so need to adjust all_logprobs_list
                    # all_logprobs_list[z] = all_logprobs_list[z] - all_scores_list_normalizer.squeeze(1)  # bs and bs,1 -> bs
                else:
                    all_logprobs_list[z] = all_logprobs_list[
                        z
                    ] - all_scores_list_normalizer.squeeze(
                        1
                    )  # bs and bs,1 -> bs
            # print("[marginalize] ==> z = ", z, " ||  all_scores_list[z] = ", all_scores_list[z], " || label = ", label)

        if self.task_setup_type not in [
            "text",
            "gt_text",
            "seq2seq",
        ]:  # todo - need to fix this. call only with prior scores
            for z, all_scores_list_tensor_normalized_z in enumerate(logprior_zgivenx):
                # print("all_scores_list_tensor_normalized_z : ", all_scores_list_tensor_normalized_z.size()) #bs
                # print("label : ", label.size()) #bs
                for val, labeli in zip(
                    all_scores_list_tensor_normalized_z.data.cpu().numpy(),
                    label.cpu().data.numpy(),
                ):
                    self._z_class_wprob[z][labeli](math.exp(float(val)))

        # prob(y|x) = sum_z (p(z)*p(y|x,z)*p(w=1|x,z)) = \logsumexp_z[ log p(z) + log p(y\x,z) + ..  ]
        # print("[marginalize] all_logprobs_list = ", all_logprobs_list)
        # log p(y) = log \sum_z p(z)p(y|z) = = log \sum_z (\exp \log (p(z)p(y|z)) )
        # = logsumexp([\log (p(z=1)p(y|z=1)),\log (p(z=2)p(y|z=2)),..])
        all_logprobs = torch.cat(all_logprobs_list, dim=0)
        # all_logprobs_list: num_programs, bs, num_labels or num_programs, bs
        total_logprob = torch.logsumexp(
            all_logprobs, dim=0
        )  # all_logprobs: bs, num_labels or bs
        # print("[marginalize] all_logprobs= ", all_logprobs)
        # print("[marginalize] total_logprob= ", total_logprob)

        mutual_kl_distance = 0.0
        approx_posterior_batchavg_entropy = 0.0
        pseudo_label_loss_value = None
        if self.use_inference_network:  # 3 need to add kl term
            # all_scores_list_tensor_normalized -> Z, bs
            # approx_posterior -> bs,Z
            # compute KL(q||p)
            # Just as before, use approx_posterior_attached (as oposed to approx_posterior_detached) since we want gradient to flow through this
            kl_distance = self.inference_network.compute_kl_distance(
                log_posterior=approx_posterior_attached, log_prior=logprior_zgivenx.t()
            )  # kl_distance: bs
            if self.kl_threshold is not None:
                kl_distance = torch.max(
                    kl_distance, kl_distance * 0.0 + self.kl_threshold
                )
            # kl_weight = 1.0
            if self.training:
                if self.kl_annealing_type is not None:
                    if self.kl_annealing_type == "linear_0_005":
                        kl_weight = self.klterm_weight = min(
                            1.0, self.klterm_weight + 0.005
                        )
                    else:
                        raise NotImplementedError
                else:
                    kl_weight = self.klterm_weight  # 0.5
            else:
                kl_weight = 1.0
            # Now add the reconstruction and the KL terms
            total_logprob = total_logprob - kl_weight * kl_distance
            # Separately track the kl term
            self._kl_distance_tracker(torch.mean(kl_distance).data.cpu().item())

            approx_posterior_batchavg_attached = (
                approx_posterior_batchavg_attached.unsqueeze(0)
            )  # Z -> 1,Z
            approx_posterior_batchavg_entropy = self.prior_model.entropy(
                approx_posterior_batchavg_attached
            )
            self._posterior_batchavg_entropy_tracker(
                approx_posterior_batchavg_entropy.mean().data.cpu().item()
            )  #

            # if self.usual_mutual_diversity:
            # approx_posterior_attached: bs, Z
            bs = approx_posterior_attached.size()[0]
            approx_posterior_attached_z1 = approx_posterior_attached[: bs // 2]
            approx_posterior_attached_z2 = approx_posterior_attached[bs // 2 :]
            # todo - are these log probs or probs ? -> log probs
            # print("approx_posterior_attached_z1 : ", approx_posterior_attached_z1.size())
            # print("approx_posterior_attached_z2 : ", approx_posterior_attached_z2.size())
            mutual_kl_distance = self.inference_network.compute_kl_distance(
                log_posterior=approx_posterior_attached_z1,
                log_prior=approx_posterior_attached_z2,
            )
            # mutual_kl_distance: bs/2
            mutual_kl_distance = mutual_kl_distance.mean()
            if self.mutual_kl_distance_use_log:
                mutual_kl_distance = torch.log(1.0 + torch.exp(-mutual_kl_distance))
                # this reverses the directionality though
                # so changge weight accordingly
            self._mutualkl_distance_tracker(mutual_kl_distance.data.cpu().item())
            # print("mutual_kl_distance = ", mutual_kl_distance)

            if heuristic_label is not None:
                # self._pseudo_label_loss = torch.nn.CrossEntropyLoss(ignore_index=0)
                # print(approx_posterior_vals['logits'].size())
                # print(heuristic_label.size())
                pseudo_label_loss_value = self._pseudo_label_loss(
                    approx_posterior_vals["logits"], heuristic_label
                )
                # 1) heuristic_label - inference network label. ignore wheh label is 0
                # print("pseudo_label_loss_value = ", pseudo_label_loss_value, " || heuristic_label = ", heuristic_label)
                self._pseudo_label_loss_value_tracker(
                    pseudo_label_loss_value.mean().data.cpu().item()
                )


        if self.task_setup_type in ["text"]:
            avg_loss_for_ppl = torch.mean(-total_logprob)
            self.text_model.log_ppl(avg_loss_for_ppl)

        output_dict = {
            "total_logprob": total_logprob,
            "all_logprobs": all_logprobs.data.cpu().numpy(),
            "loss_prior": None,
            "approx_posterior_entropy": approx_posterior_entropy,
            "approx_posterior_batchavg_entropy": approx_posterior_batchavg_entropy,
            "prior_entropy": prior_entropy,
            "prior_batchavg_entropy": prior_batchavg_entropy,
            "mutual_kl_distance": mutual_kl_distance,
            "pseudo_label_loss_value": pseudo_label_loss_value,
            "all_scores_list_tensor_l2": all_scores_list_tensor_l2,
            # 'analysis_locate':analysis_vals['locate'],
            # 'analysis_attend':analysis_vals['attend'],
            # 'analysis_combine':analysis_vals['combine_v3']
        }

        # if self.use_inference_network:
        #     output_dict['inferencenw_text_used'] = approx_posterior_vals['raw_text']
        #     output_dict['inferencenw_loss'] = approx_posterior_vals['loss']

        return output_dict

    # inference
    # condition on label text (and optionally the series)
    # get approx posterior on the program distr.
    # for now, sum over
    # KL: prior needs to be computed
    # so stesps
    # 1) get approx posterior
    # 2) marginalize over the approx posterior to compute conditional
    # 3) KL loss computation
    # p(z|x) \propto exp( f(z) * score(z,x) ). currently f(z) is not used.
    # q(z|x,y) ...
    # when f(z) is not used, eaxct KL computation would involve computing prior
    # -- add a function to compute prior. will use in marginalize and KL
    # -- add a function to marginalize over a given distribution. will use in inference and marginalize_new
    # inference network:
    # -> may be lstm or mean embedding for now
    # test it out by trying to predict [1] class  [2] pattern and location both


    def sample(
        self,
        series,
        label,
        label_text,
        predictor_network,
        use_argmax: bool = False,
        metadata=None,
        num_samples=1,
    ):

        vals = predictor_network(series)
        logits = vals["logits"]  # bs, program_space
        bs = series.size()[0]

        dist: torch.distributions.Categorical = torch.distributions.Categorical(
            logits=logits
        )
        all_logprobs_list = []
        logprob_z_list = []

        # for z in range(self.num_programs):
        #     print("[sample] [NewThreeLayerLayoutPredictionProgram]: ", z, self.programs[z].__str__)
        #     print("[sample] **Prior Distribution z=", z, torch.exp(dist.log_prob(torch.tensor(z))))
        # print()
        # print("[sample] label = ", label)
        # print()

        const = torch.ones(bs)
        if torch.cuda.is_available():
            const = const.cuda()
        const = num_samples * const  # bs. all values are equal to K (num_samples)

        for k in range(num_samples):

            ##### sampling / argmax
            if use_argmax:
                action, z_embedding, logprob_z = predictor_network.get_argmax(logits)
                assert num_samples == 1
                # print("[sample] mode = argmax || action = ", action)
            else:
                action, z_embedding, logprob_z = predictor_network.sample(logits)
            logprob_z_list.append(logprob_z)

            zlist = list(action.data.cpu().numpy())  # bs
            # print("[sample] zlist = ", zlist)
            # print("[sample] logprob_z = ", logprob_z)
            # print("[sample] series : ", series.size())
            for zz in zlist:
                self._z_chosen_counts[zz] += 1

            ##### compute program
            program_list = [self.programs[z] for z in zlist]
            logprob_w_given_z_list = [
                torch.log(programj(series[j : j + 1]))
                for j, programj in enumerate(program_list)
            ]
            logprob_w_given_z = torch.stack(logprob_w_given_z_list)  # bs,1
            # print("[sample] ==> z = ", zlist, " ||  logprob_w_given_z = ", logprob_w_given_z, " \\\ label = ", label)
            if self.task_setup_type != "text":
                for zz, val, labelb in zip(
                    zlist,
                    logprob_w_given_z.cpu().data.numpy(),
                    label.cpu().data.numpy(),
                ):
                    self._z_class_wprob[zz][labelb](val)

            ##### get output prob. given program
            logprob_y_given_z_k, _ = self.get_proby_givenz(
                z_embedding=z_embedding,
                label=label,
                label_text=label_text,
                metadata=metadata,
                selected_program_id=action.data.cpu().item(),
            )

            #### computing cur_prob_k
            cur_prob_k = logprob_w_given_z + logprob_y_given_z_k
            all_logprobs_list.append(cur_prob_k.unsqueeze(0))

        # logprob_y_given_z_numpy = logprob_y_given_z.data.cpu().numpy()
        # assert len(logprob_y_given_z_numpy.shape)==2, logprob_y_given_z_numpy
        # for zz,logprob_y_given_z_numpy_zidx in zip(zlist,logprob_y_given_z_numpy):
        #     for j,val in enumerate(logprob_y_given_z_numpy_zidx):
        #         self._z_class_probs[zz][j](val)
        # print("[sample] ==> zlist = ", zlist, " ||  logprob_y_given_z = ", logprob_y_given_z)

        # cur_prob = logprob_z.unsqueeze(1) + logprob_w_given_z.unsqueeze(1) + logprob_y_given_z # bs, num_labels
        # logprob_y_w1_given_z = logprob_w_given_z + logprob_y_given_z # bs, num_labels
        # print("[sample] ==> z = ", zlist, " ||  logprob_y_w1_given_z = ", logprob_y_w1_given_z)
        # print()

        all_logprobs = torch.cat(
            all_logprobs_list, dim=0
        )  # all_logprobs_list: num_programs, bs, num_labels
        total_logprob = torch.logsumexp(
            all_logprobs, dim=0
        )  # total_logprob: bs, num_labels
        # print("[sample]: total_logprob  : ", total_logprob.size())
        total_logprob = total_logprob - torch.log(const).unsqueeze(
            1
        )  # subtract logK to normalize
        # print("[sample] total_logprob= ", total_logprob)

        # all_logprobs_list: num_samples, bs, num_labels
        # logprob_z_list: num_samples, bs
        logprob_z_list = torch.cat(logprob_z_list, dim=0)

        output_dict = {
            "total_logprob": total_logprob,
            "logprob_z": logprob_z_list,
            "all_logprobs_list": all_logprobs,
            "type": "multiple_samples",
        }

        return output_dict


    def gt_program_text(self, label_text, predictor_network, metadata, mode="ppl"):

        # bs = label_text['tokens'].size()[0]
        labels = [metadata_i["label"]["labels"] for metadata_i in metadata]
        if self.num_programs == 6:
            mapper = LABEL_TO_PROGRAM_MAPPER_6programs
        elif self.num_programs == 12:
            mapper = LABEL_TO_PROGRAM_MAPPER_12programs
        else:
            raise NotImplementedError
        gt_programs = [mapper[label] for label in labels]
        gt_programs = np.array(gt_programs, dtype=np.long)
        gt_programs = torch.tensor(gt_programs)
        if torch.cuda.is_available():
            gt_programs = gt_programs.cuda()
        # print("[gt_program_text] labels = ", labels)
        # print("[gt_program_text] label_text = ", label_text)
        # print("[gt_program_text] gt_programs = ", gt_programs)
        # print("[gt_program_text] gt_programs = ", [self.programs[g].__str__ for g in gt_programs.data.cpu().numpy()])
        z_embedding = predictor_network.get_program_emb(gt_programs)

        logprob_y, vals = self.get_proby_givenz(
            z_embedding=z_embedding,
            label=None,
            label_text=label_text,
            metadata=metadata,
            mode=mode,
        )
        total_logprob = logprob_y

        output_dict = {"total_logprob": total_logprob, "type": "gt_program_text"}
        if mode == "generate":
            output_dict = {"generated_" + k: val for k, val in output_dict.items()}

        return output_dict

    def seq2seq_generate(self, series, label_text, metadata, mode="generate"):

        assert mode == "generate"
        z_embedding = self.seq2seq_encoder.forward(series)

        logprob_y, vals = self.get_proby_givenz(
            z_embedding=z_embedding,
            label=None,
            label_text=label_text,
            metadata=metadata,
            mode=mode,
        )
        total_logprob = logprob_y
        output_dict = {"total_logprob": total_logprob, "type": "seq2seq"}
        if mode == "generate":
            output_dict = {"generated_" + k: val for k, val in output_dict.items()}
            if not self.training:
                output_dict.update(
                    {
                        "generate_all_generated": vals["generate_all_generated"],
                        "generate_all_target": vals["generate_all_target"],
                    }
                    # 'locate': v1,
                    # 'attend': v2,
                    # 'combine_v3': v3,
                    # 'combine_score': score
                )
        return output_dict

    def seq2seq(
        self,
        series,
        label_text,
        metadata,
        mode="ppl",
        heuristic_label=None
    ):

        # bs = label_text['tokens'].size()[0]
        # labels = [metadata_i['label']['labels'] for metadata_i in metadata]
        # labels = [metadata_i['label'] for metadata_i in metadata]
        # labels = [label['labels'] if 'labels' in label else label for label in labels]

        if self.num_programs != 0:
            print("self.num_programs = ", self.num_programs)
            raise NotImplementedError
        #     mapper = LABEL_TO_PROGRAM_MAPPER_6programs
        # elif self.num_programs == 12:
        #     mapper = LABEL_TO_PROGRAM_MAPPER_12programs
        # else:
        #     raise NotImplementedError
        encoded_series = self.seq2seq_encoder.forward(series)

        logprob_y, vals = self.get_proby_givenz(
            z_embedding=encoded_series,
            label=None,
            label_text=label_text,
            metadata=metadata,
            mode=mode,
        )
        total_logprob = logprob_y
        output_dict = {"total_logprob": total_logprob, "type": "gt_program_text"}
        self.text_model.log_ppl(-total_logprob.mean())

        if self.seq2seq_use_multitask:
            logits = self.seq2seq_multitask_predictor(
                encoded_series
            )  # encoded_series size -> num_labels
            # print("logits: ", logits.size())
            # print("heuristic_label: ", heuristic_label.size(), "heuristic_label = ", heuristic_label)
            multitask_label_loss = self._pseudo_label_loss(logits, heuristic_label)
            output_dict["multitask_label_loss"] = multitask_label_loss
            # print("multitask_label_loss = ", multitask_label_loss)
            self._multitask_heuristic_loss_tracker(
                multitask_label_loss.data.cpu().item()
            )

        if mode == "generate":
            output_dict = {"generated_" + k: val for k, val in output_dict.items()}

        return output_dict


    def generate(
        self,
        series,
        label,
        label_text,
        predictor_network,
        use_argmax: bool = False,
        program_selection_method: str = "argmax",
        program_selection_method_topk: int = 10,
        metadata=None,
        use_prior_potential: bool = True,
    ):

        bs = series.size()[0]
        assert self.task_setup_type == "text"

        vals = predictor_network(series)
        logits = vals["logits"]  # bs, program_space
        dist: torch.distributions.Categorical = torch.distributions.Categorical(
            logits=logits
        )

        all_logprobs_list = []
        all_scores_list = []

        # for z in range(self.num_programs):
        #     print("[marginalize] [NewThreeLayerLayoutPredictionProgram]: ", z, self.programs[z].__str__)
        #     print("[marginalize] **Prior Distribution z=", z, torch.exp(dist.log_prob(torch.tensor(z))))
        # print()
        # print("label = ", label)
        # print()

        analysis_vals = {"locate": [], "attend": [], "attendpre": [], "combine_v3": []}

        for z in range(self.num_programs):

            action = torch.tensor(z)
            if torch.cuda.is_available():
                action = action.cuda()
            program = self.programs[z]
            program_ret = program.forward(
                series, get_score_also=True, analysis_mode=True
            )
            _, score_z_x = program_ret["ret"]
            all_scores_list.append(score_z_x.unsqueeze(0))
            # 'locate': v1,
            # 'attend': v2,
            # 'combine_v3': v3,
            # 'combine_score': score,
            # 'ret': ret
            analysis_vals["locate"].append(program_ret["locate"].data.cpu().numpy())
            analysis_vals["attend"].append(program_ret["attend"].data.cpu().numpy())
            analysis_vals["attendpre"].append(
                program_ret["attendpre"].data.cpu().numpy()
            )
            analysis_vals["combine_v3"].append(
                program_ret["combine_v3"].data.cpu().numpy()
            )

            # print("[generate] ==> z = ", z, " ||  score_z_x = ", score_z_x,
            #       " ||| label = ", label,
            #       " ||| label_text=", label_text)

            # cur_prob = logprob_z.unsqueeze(1) + logprob_w_given_z.unsqueeze(1) + logprob_y_given_z # bs, num_labels
            unnormalized_logprobz_givenx = score_z_x  # bs
            cur_prob_score = unnormalized_logprobz_givenx  # bs
            if use_prior_potential:
                logprob_z = dist.log_prob(action)  # bs
                # print("[generate] ==> z = ", z, "logprob_z = ", logprob_z, " ||| label = ", label)
                cur_prob_score += logprob_z  # bs

            # print("[generate] ==> z = ", z, " ||  cur_prob = ", cur_prob_score)
            all_logprobs_list.append(cur_prob_score.unsqueeze(0))  # 1, bs
            # all_logprobs_list.append(cur_prob.data.cpu().numpy())
            # print()

        # all_scores_list: numprograms,bs
        all_scores_list_tensor = torch.cat(all_scores_list, dim=0)  # nhum_programs, bs

        # -- now pick the one highest value
        # -- get corresponding z_embeddings
        # -- call get_proby_givenz with corresponding values - under 'generate' mode
        # --

        selected_program_id = None
        if program_selection_method == "argmax":
            max_scores, z_argmax = all_scores_list_tensor.max(
                dim=0
            )  # all_scores_list_tensor is programs, bs
            selected_program_id = z_argmax.data.cpu().numpy()
        elif program_selection_method == "sample":
            all_scores_list_tensor_bs_programs = all_scores_list_tensor.t()
            temperature = 1.0  # self.
            top_k = program_selection_method_topk  # 10
            top_p = 1.0  # 0.9
            bs = all_scores_list_tensor_bs_programs.size()[0]
            assert bs == 1, "found bs = " + str(bs) + " , but wanted bs = 1"
            # print("output_projections : ", output_projections.size())
            logits = all_scores_list_tensor_bs_programs.view(-1) / temperature
            filtered_logits = utils.top_k_top_p_filtering(
                logits, top_k=top_k, top_p=top_p
            )
            probabilities = F.softmax(filtered_logits, dim=-1)
            selected_program_id_tensor = z_argmax = torch.multinomial(probabilities, 1)
            selected_program_id = selected_program_id_tensor.data.cpu().numpy()
        else:
            raise NotImplementedError
        print("[generate]: z_argmax = ", z_argmax)
        print("[generate]: all_scores_list_tensor = ", all_scores_list_tensor)
        # for programi in self.programs:
        #     print(programi.locate.operator_name, programi.attend.operator_name )
        z_embedding = predictor_network.get_program_emb(z_argmax)  # bs,emb_dim
        logprob_y_given_z, generate_vals = self.get_proby_givenz(
            z_embedding=z_embedding,
            label=label,
            label_text=label_text,
            metadata=metadata,
            mode="generate",
            selected_program_id=z_argmax.data.cpu().numpy(),
        )  # logprob_y_given_z: bs
        ######

        # print("--> selected_program_id = ", selected_program_id)
        # assert len(selected_program_id) == 1
        analysis_vals_all = analysis_vals
        analysis_vals = {k: v[selected_program_id[0]] for k, v in analysis_vals.items()}
        ######
        all_scores_list_normalizer = torch.logsumexp(
            all_scores_list_tensor, dim=0
        )  # bs
        all_scores_list_normalizer = all_scores_list_normalizer  # bs
        logprob_y_given_z = (
            logprob_y_given_z - all_scores_list_normalizer
        )  ### TODO -- why ??
        # logprob_z = all_scores_list_tensor - all_scores_list_normalizer
        # all_scores_list_tensor -> programs, bs :: all_scores_list_tensor.t() -> bs,programs
        all_scores_list_tensor_str = [
            "_".join([str(y) for y in yy])
            for yy in all_scores_list_tensor.t().data.cpu().numpy()
        ]
        # all_scores_list_tensor_str = '_'.join(all_scores_list_tensor_str)

        output_dict = {
            "generate_total_logprob": logprob_y_given_z,
            "generate_z_argmax": z_argmax.data.cpu().numpy(),
            "generate_z_argmax_scores": all_scores_list_tensor_str,  # max_scores.data.cpu().numpy(),
            "generate_z_argmax_programname": [
                self.programs[zx].__str__ for zx in z_argmax.data.cpu().numpy()
            ],
        }
        # output_dict.update({'generate_all_generated': all_generated, 'generate_all_target': all_target})
        if not self.training:
            # print("generate_vals : ", generate_vals.keys())
            output_dict.update(
                {
                    "generate_all_generated": generate_vals["generate_all_generated"],
                    "generate_all_target": generate_vals["generate_all_target"],
                    "analysis_locate": [
                        "_".join([str(y) for y in analysis_vals["locate"][0]])
                    ],
                    "analysis_attend": [
                        "_".join([str(y) for y in analysis_vals["attend"][0]])
                    ],
                    "analysis_attendpre": [
                        "_".join([str(y) for y in analysis_vals["attendpre"][0]])
                    ],
                    "analysis_combine": [str(analysis_vals["combine_v3"][0])],
                }
            )
            # print("selected_program_id = ", selected_program_id)
            if len(selected_program_id) == 1:
                # print("analysis_vals_all : ", analysis_vals_all['locate'])
                # print("analysis_vals_all : ", len(analysis_vals_all['locate']) )
                # print("analysis_vals_all : ", len(analysis_vals_all['locate'][0]) )
                # 0/0
                output_dict.update(
                    {
                        "analysis_vals_all_locate": [analysis_vals_all["locate"]],
                        "analysis_vals_all_attend": [analysis_vals_all["attend"]],
                    }
                )
        return output_dict


    def text_evals(self, label_text, tmp, output_dict):
        total_logprob = tmp["total_logprob"]
        # print(" *********************** total_logprob = ", total_logprob)
        loss = -torch.mean(total_logprob)
        if self.use_inference_network:
            output_dict["elbo_loss"] = loss
            # following two lines needed if using reinforce
            # output_dict["loss_prior"] = tmp['loss_prior']
            # loss = loss + self.reinforce_loss_coef * tmp['loss_prior']
            output_dict["approx_posterior_entropy"] = tmp["approx_posterior_entropy"]
            output_dict["approx_posterior_batchavg_entropy"] = tmp[
                "approx_posterior_batchavg_entropy"
            ]
            # output_dict['prior_entropy'] = tmp['prior_entropy']
            if self.inferencenw_entropy_wt > 0:
                loss = (
                    loss - self.inferencenw_entropy_wt * tmp["approx_posterior_entropy"]
                )  # penalize low entropy.
            if self.inferencenw_batchavg_entropy_wt > 0:
                loss = (
                    loss
                    - self.inferencenw_batchavg_entropy_wt
                    * tmp["approx_posterior_batchavg_entropy"]
                )  # penalize low entropy.
            # elbo_loss_tracking = loss_lm - todo
            if self.mutual_kl_distance_use_log:
                # however, when using the log(1+exp(-mut_kl)), then use '+'
                loss = (
                    loss + self.mutual_kl_distance_wt * tmp["mutual_kl_distance"]
                )  # penalize low mutual_kl_distance.
            else:
                loss = (
                    loss - self.mutual_kl_distance_wt * tmp["mutual_kl_distance"]
                )  # penalize low mutual_kl_distance.
            if tmp["pseudo_label_loss_value"] is not None:
                loss = loss + tmp["pseudo_label_loss_value"]
            if self.kl_sanity_check:
                loss = tmp["pseudo_label_loss_value"]
        if self.task_setup_type in ['text']:
            if self.use_all_scores_list_tensor_l2:
                loss = loss + self.program_score_l2_wt * tmp["all_scores_list_tensor_l2"]
            if self.use_prior_entropy_regularize:
                loss = (
                    loss - self.prior_entropy_wt * tmp["prior_entropy"]
                )  # penalize low entropy
            if self._prior_batchavg_entropy_wt > 0.0:
                print("tmp['prior_batchavg_entropy'] = ", tmp["prior_batchavg_entropy"])
                loss = (
                    loss - self._prior_batchavg_entropy_wt * tmp["prior_batchavg_entropy"]
                )  # penalize low entropy
        if self.seq2seq_use_multitask:
            # 'multitask_label_loss' in tmp:
            # print(tmp['multitask_label_loss'].size())
            # print(loss.size())
            loss = loss + self.seq2seq_multitask_loss_wt * tmp["multitask_label_loss"]
        output_dict["loss"] = loss
        if not self.training:
            output_dict.update(tmp)
        return output_dict

    def forward(
        self,
        series: torch.FloatTensor,
        label: torch.LongTensor = None,
        label_text: [str, torch.LongTensor] = None,
        feats=None,
        distractors=None,
        metadata: List[Dict[str, Any]] = None,
        heuristic_label=None
    ):

        predictor_network: EnumerateAllPrior = self.prior_model
        bs = series.size()[0]

        # print("label_text = ", label_text)
        # print("label = ", label)

        if self.task_setup_type == "classification":
            pass
        else:
            pass
            # targets = label_text["tokens"]
            # target_mask = get_text_field_mask(label_text)

        tmp = {}

        if self.training:

            if self.task_setup_type == "gt_program_text":
                tmp.update(
                    self.gt_program_text(
                        label_text=label_text,
                        metadata=metadata,
                        predictor_network=predictor_network,
                    )
                )

            elif self.task_setup_type == "seq2seq":
                tmp.update(
                    self.seq2seq(
                        series=series,
                        label_text=label_text,
                        metadata=metadata,
                        heuristic_label=heuristic_label
                    )
                )

            elif self.task_setup_type == "classification_seq2seq":
                tmp.update(
                    self.seq2seq(
                        series=series,
                        label_text=label_text,
                        metadata=metadata,
                        classification_seq2seq=True,
                    )
                )

            else:

                if self.model_type == "marginalize":
                    tmp.update(
                        self.marginalize(
                            series,
                            label,
                            label_text,
                            predictor_network,
                            metadata=metadata,
                        )
                    )

                elif self.model_type == "prior_reinforce":
                    # compute predictor loss
                    tmp.update(
                        self.sample(
                            series,
                            label,
                            label_text,
                            predictor_network,
                            metadata=metadata,
                            num_samples=self.reinforce_num_samples,
                        )
                    )
                    # get reward
                    # compute prior loss
                    # prior loss. add to total

                elif self.model_type == "inference_nw":
                    raise NotImplementedError
                    # get reward. this updates the posterior. add to total loss
                    # get kl b.w prior and posteriror. add to total loss. this updates both prior and inference

                elif self.model_type == "marginalize_new":
                    tmp.update(
                        self.marginalize_new(
                            series,
                            label,
                            label_text,
                            predictor_network,
                            metadata=metadata,
                            train_unary_prior=self.train_unary_prior,
                            use_prior_potential=self.use_prior_potential,
                            heuristic_label=heuristic_label
                        )
                    )

        else:

            # test_time_method_type

            if self.task_setup_type == "gt_program_text":
                tmp.update(
                    self.gt_program_text(
                        label_text=label_text,
                        metadata=metadata,
                        predictor_network=predictor_network,
                    )
                )
                generate_vals = self.gt_program_text(
                    label_text=label_text,
                    metadata=metadata,
                    predictor_network=predictor_network,
                    mode="generate",
                )
                for k in generate_vals:
                    assert k not in tmp
                tmp.update(generate_vals)

            elif self.task_setup_type == "seq2seq":
                tmp.update(
                    self.seq2seq(
                        series=series,
                        label_text=label_text,
                        metadata=metadata,
                        heuristic_label=heuristic_label
                    )
                )
                generate_vals = self.seq2seq_generate(
                    series=series, label_text=label_text, metadata=metadata
                )
                print("---> generate_vals = ", generate_vals)
                for k in generate_vals:
                    assert k not in tmp
                tmp.update(generate_vals)

            elif self.task_setup_type == "classification_seq2seq":
                tmp.update(
                    self.seq2seq(
                        series=series, label_text=label_text, metadata=metadata
                    )
                )

            else:

                if self.test_time_method_type == "marginalize":
                    tmp.update(
                        self.marginalize(
                            series,
                            label,
                            label_text,
                            predictor_network,
                            metadata=metadata,
                        )
                    )

                elif self.test_time_method_type == "sample":
                    assert False  # not updated for latest changes in prior model
                    # tmp.update(self.sample(series, label, label_text, predictor_network, metadata=metadata))

                elif self.test_time_method_type == "marginalize_new":
                    tmp.update(
                        self.marginalize_new(
                            series,
                            label,
                            label_text,
                            predictor_network,
                            metadata=metadata,
                            train_unary_prior=self.train_unary_prior,
                            use_prior_potential=self.use_prior_potential,
                            heuristic_label=heuristic_label
                        )
                    )

                elif self.test_time_method_type == "argmax":
                    assert False  # not updated for latest changes in prior model
                    # tmp.update(self.sample(series, label, label_text,
                    #                        predictor_network,
                    #                        use_argmax=True,
                    #                        metadata=metadata,
                    #                        num_samples=1))

                if self.task_setup_type in ["text"]:  # ,'seq2seq']:
                    generate_vals = self.generate(
                        series,
                        label,
                        label_text,
                        predictor_network,
                        metadata=metadata,
                        program_selection_method=self.program_selection_method,
                        program_selection_method_topk=self.program_selection_method_topk,
                    )
                    for k in generate_vals:
                        assert k not in tmp
                    tmp.update(generate_vals)

        # print(">>>>>> metadata = >>>>>> ", metadata)

        output_dict = {}

        if (
            self.task_setup_type in ["classification", "classification_seq2seq"]
            and label is not None
        ):
            total_logprob = tmp["total_logprob"]
            tmp.update({"bs": bs})
            output_dict = self.classification_evals(
                label=label, tmp=tmp, output_dict=output_dict
            )
        elif (
            self.task_setup_type
            in ["text", "unconditional_lm", "gt_program_text", "seq2seq"]
            and label_text is not None
        ):
            output_dict = self.text_evals(
                label_text=label_text, tmp=tmp, output_dict=output_dict
            )
            # print("metadata = ", metadata)
            # output_dict.update({k:[v] for k,v in metadata[0].items()})
            output_dict.update(
                {
                    k: [metadataj[k] for metadataj in metadata]
                    for k, v in metadata[0].items()
                }
            )
            # output_dict.update({'series':[list(series.data.cpu().numpy()[0])]} )
            output_dict.update({"series": [list(series.data.cpu().numpy())]})
            # total_logprob = tmp['total_logprob']
            # output_dict = self.classification_evals(total_logprob, output_dict)
        else:
            raise NotImplementedError

        # print("--->> outputs ----> ", output_dict.keys())
        if self.simplify_outputs:
            output_dict = {
                k: v
                for k, v in output_dict.items()
                if k
                in [
                    "idx",
                    "series",
                    "generate_all_generated",
                    "generate_all_target",
                    "generate_z_argmax",
                    "generate_z_argmax_scores",
                    "generate_z_argmax_programname",
                    "analysis_locate",
                    "analysis_attend",
                    "analysis_attendpre",
                    "analysis_combine",
                    "analysis_vals_all_locate",
                    "analysis_vals_all_attend",
                ]
            }
            # print("poutputs --> ", output_dict)
            # 'generate_all_generated', 'generate_all_target', 'col_names', 'idx', 'raw_text', 'series'
            # print("output_dict['idx'] = ",output_dict['idx'])
            # # print("output_dict['loss'] = ",output_dict['loss'])
            # print("output_dict['series'] = ",output_dict['series'])
            # print("output_dict['generate_all_generated'] = ",output_dict['generate_all_generated'])
            # print("output_dict['generate_all_target'] = ",output_dict['generate_all_target'])

        if self.use_inferencenw_evals:
            if self.model_name is not None:
                json.dump(
                    self.inference_network.use_inferencenw_evals_outputs,
                    open(
                        "tmp/"
                        + self.model_name
                        + "/use_inferencenw_evals_outputs.json",
                        "w",
                    ),
                )

        # print("output_dict = ", output_dict)

        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        metrics = {}

        if self.task_setup_type in [
            "text",
            "unconditional_lm",
            "gt_program_text",
            "seq2seq",
        ]:

            metrics.update(self.text_model.get_metrics(reset))
            metrics.update(
                {"prior_entropy": self._prior_entropy_tracker.get_metric(reset)}
            )
            metrics.update(
                {
                    "prior_batchavg_entropy": self._prior_batchavg_entropy_tracker.get_metric(
                        reset
                    )
                }
            )
            metrics.update(
                {"scores_l2_tracker": self.scores_l2_tracker.get_metric(reset)}
            )
            if self.use_inference_network:
                metrics.update(
                    {
                        "inferencenw_kl_distance": self._kl_distance_tracker.get_metric(
                            reset
                        )
                    }
                )
                metrics.update(
                    {
                        "inferencenw_mutualkl_dist": self._mutualkl_distance_tracker.get_metric(
                            reset
                        )
                    }
                )
                metrics.update(
                    {
                        "inferencenw_priorloss": self._prior_loss_tracker.get_metric(
                            reset
                        )
                    }
                )
                metrics.update(
                    {"inferencenw_elboloss": self._elbo_tracker.get_metric(reset)}
                )
                metrics.update(
                    {"inferencenw_reward": self._reward_tracker.get_metric(reset)}
                )
                metrics.update(
                    {
                        "posterior_batchavg_entropy": self._posterior_batchavg_entropy_tracker.get_metric(
                            reset
                        )
                    }
                )
                metrics.update(
                    {
                        "inferencenw_" + k: v
                        for k, v in self.inference_network.get_metrics(reset).items()
                    }
                )
                metrics.update(
                    {
                        "inferencenw_pseudolabelloss_": self._pseudo_label_loss_value_tracker.get_metric(
                            reset
                        )
                    }
                )
            if self.seq2seq_use_multitask:
                metrics.update(
                    {
                        "multitask_heuristic_loss": self._multitask_heuristic_loss_tracker.get_metric(
                            reset
                        )
                    }
                )
            if self.use_bertscore_evals:
                pass
                # info = test_multi_refs_working(self.bert_scorer,
                #                                bertscore_cands=self.bertscore_cands,
                #                                bertscore_refs=self.bertscore_refs)
                # self._bert_p_r_f1_tracker

        # if self.program_trainable_locate:
        if self.task_setup_type in ['text']:
            for locate_inst in self.instances["locate"]:
                metrics.update(locate_inst.get_useful_partitions())
            # if self.program_trainable_attend:
            for inst in self.instances["attend"]:
                metrics.update(inst.get_useful_partitions())

        return metrics







from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
import json
from overrides import overrides
from allennlp.data import DatasetReader, Instance
from allennlp.common.util import JsonDict, sanitize



@Predictor.register("ns_stock_predictor")
class NSStockPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._model = model
        self._cnt = 0
        self._fname_to_dump = fname_to_dump = (
            model.predictor_file_name
            if model.predictor_file_name is not None
            else "tmp/" + model.model_name + "/predictions.tsv"
        )
        self.print_moreinfo = True
        fw = open(fname_to_dump, "w")
        vals = ["idx", "series", "one-of-the-references", "generated text"]
        if self.print_moreinfo:
            vals.extend(
                [
                    "program_selected",
                    "program_selected_modules",
                    "program_selected_scores",
                ]
            )
            vals.extend(
                [
                    "analysis_attend",
                    "analysis_locate",
                    "analysis_combine",
                    "analysis_attendpre",
                ]
            )
        fw.write(",".join(vals))
        fw.write("\n")
        fw.close()

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        self._cnt += 1
        # {"loss": 1.122908115386963, "total_logprob": -1.122908115386963, "generate_total_logprob": -4.646208763122559,
        #  "generate_z_argmax": 0, "generate_all_generated": "decreases at the beginning",
        #  "generate_all_target": "dips at the start"}
        fw = open(self._fname_to_dump, "a")
        print("--->>>> outputs = ", outputs)
        outputs = outputs[0]
        print("--->>>> outputs = ", outputs.keys())
        vals = [
            str(outputs["idx"]),
            str(outputs["series"]).replace(",", " "),
            outputs["generate_all_target"],
            outputs["generate_all_generated"],
        ]
        if self.print_moreinfo:
            vals.extend(
                [
                    str(outputs["generate_z_argmax"]),
                    outputs["generate_z_argmax_programname"],
                    str(outputs["generate_z_argmax_scores"]),
                ]
            )
            vals.extend(
                [
                    str(outputs["analysis_attend"]),
                    str(outputs["analysis_locate"]),
                    str(outputs["analysis_combine"]),
                    str(outputs["analysis_attendpre"]).replace("\n", " "),
                ]
            )
        fw.write(",".join(vals))
        fw.write("\n")
        fw.close()
        return json.dumps(outputs) + "\n"

    def predict_instance(self, instance: Instance) -> JsonDict:
        print("--- instance = ", instance)
        # outputs = self._model.forward_on_instance(instance)
        outputs = self._model.forward_on_instances([instance])
        return sanitize(outputs)


@Predictor.register("ns_predictor")
class NSPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._model = model
        self._cnt = 0
        self._fname_to_dump = fname_to_dump = (
            model.predictor_file_name
            if model.predictor_file_name is not None
            else "tmp/" + model.model_name + "/predictions.tsv"
        )
        self.print_moreinfo = False  # True #False
        fw = open(fname_to_dump, "w")
        vals = ["idx", "series", "GT-label", "one-of-the-references", "generated text"]
        if self.print_moreinfo:
            vals.extend(
                [
                    "program_selected",
                    "program_selected_modules",
                    "program_selected_scores",
                ]
            )
            vals.extend(
                [
                    "analysis_attend",
                    "analysis_locate",
                    "analysis_combine",
                    "analysis_attendpre",
                ]
            )
            # vals = ['idx', 'series', 'GT-label','one-of-the-references','generated text','module-list','program_selected']
            # vals.extend(['analysis_attend', 'analysis_locate', 'analysis_combine', 'analysis_attendpre' ])
        fw.write(",".join(vals))
        fw.write("\n")
        fw.close()

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        self._cnt += 1
        # {"loss": 1.122908115386963, "total_logprob": -1.122908115386963, "generate_total_logprob": -4.646208763122559,
        #  "generate_z_argmax": 0, "generate_all_generated": "decreases at the beginning",
        #  "generate_all_target": "dips at the start"}
        fw = open(self._fname_to_dump, "a")
        # print("--->>>> outputs = ", outputs)
        outputs = outputs[0]
        # print("--->>>> outputs = ", outputs.keys())
        if "label" in outputs:
            vals = [
                str(outputs["idx"]),
                str(outputs["series"]).replace(",", " "),
                label_mapper[outputs["label"]["labels"]],
                outputs["generate_all_target"],
                outputs["generate_all_generated"],
            ]
        else:
            if self.print_moreinfo:
                vals = [
                    str(outputs["idx"]),
                    str(outputs["series"]).replace(",", " "),
                    "",
                    outputs["generate_all_target"],
                    outputs["generate_all_generated"],
                ]
                vals.extend(
                    [
                        str(outputs["generate_z_argmax"]),
                        outputs["generate_z_argmax_programname"],
                        str(outputs["generate_z_argmax_scores"]),
                    ]
                )
                vals.extend(
                    [
                        str(outputs["analysis_attend"]),
                        str(outputs["analysis_locate"]),
                        str(outputs["analysis_combine"]),
                        str(outputs["analysis_attendpre"]).replace("\n", " "),
                    ]
                )
            else:
                vals = [
                    str(outputs["idx"]),
                    str(outputs["series"]).replace(",", " "),
                    outputs["generate_all_target"],
                    outputs["generate_all_generated"],
                ]
        fw.write(",".join(vals))
        fw.write("\n")
        fw.close()
        return json.dumps(outputs) + "\n"

    def predict_instance(self, instance: Instance) -> JsonDict:
        print("--- instance = ", instance)

        # outputs = self._model.forward_on_instance(instance)
        outputs = self._model.forward_on_instances([instance])
        # instances = [instance]
        # batch_size = len(instances)
        # with torch.no_grad():
        #     cuda_device = self._get_prediction_device()
        #     dataset = Batch(instances)
        #     dataset.index_instances(self.vocab)
        #     model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
        #     outputs = self.make_output_human_readable(self(**model_input))
        #
        #     instance_separated_output: List[Dict[str, numpy.ndarray]] = [
        #         {} for _ in dataset.instances
        #     ]
        #     for name, output in list(outputs.items()):
        #         if isinstance(output, torch.Tensor):
        #             # NOTE(markn): This is a hack because 0-dim pytorch tensors are not iterable.
        #             # This occurs with batch size 1, because we still want to include the loss in that case.
        #             if output.dim() == 0:
        #                 output = output.unsqueeze(0)
        #             if output.size(0) != batch_size:
        #                 self._maybe_warn_for_unseparable_batches(name)
        #                 continue
        #             output = output.detach().cpu().numpy()
        #         elif len(output) != batch_size:
        #             self._maybe_warn_for_unseparable_batches(name)
        #             continue
        #         for instance_output, batch_element in zip(instance_separated_output, output):
        #             instance_output[name] = batch_element
        #     outputs = instance_separated_output
        # print("--- outputs = ", outputs)
        return sanitize(outputs)


from allennlp_series.data.dataset_reader.stock_text_reader import StockDataTextReader


@Predictor.register("ns_predictor_fromsynthpretrained")
class NSPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, StockDataTextReader())
        self._model = model
        self._cnt = 0
        self._fname_to_dump = fname_to_dump = (
            model.predictor_file_name
            if model.predictor_file_name is not None
            else "tmp/" + model.model_name + "/predictions.tsv"
        )
        fw = open(fname_to_dump, "w")
        vals = ["idx", "series", "one-of-the-references", "generated text"]
        fw.write(",".join(vals))
        fw.write("\n")
        fw.close()

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        self._cnt += 1
        # {"loss": 1.122908115386963, "total_logprob": -1.122908115386963, "generate_total_logprob": -4.646208763122559,
        #  "generate_z_argmax": 0, "generate_all_generated": "decreases at the beginning",
        #  "generate_all_target": "dips at the start"}
        fw = open(self._fname_to_dump, "a")
        print("--->>>> outputs = ", outputs.keys())
        print("--->>>> outputs = ", outputs)
        vals = [
            str(outputs["idx"]),
            str(outputs["series"]).replace(",", " "),
            outputs["generate_all_target"],
            outputs["generate_all_generated"],
        ]
        fw.write(",".join(vals))
        fw.write("\n")
        fw.close()
        return json.dumps(outputs) + "\n"



if __name__ == "__main__":
    model = TRUCEMethod(Vocabulary())
    """arr = torch.tensor([[0.0100, 0.0333, -0.0266, -0.0166, 0.0333, 0.0133, -0.0300,
                         0.0033, 0.0100, 0.0333, -0.0266, -0.0166],
                        [0.3100, 0.0333, -0.0266, -0.0166, 0.0333, 0.0133, -0.0300,
                         0.3033, 0.0100, 0.0333, -0.0266, -0.0166]])
    """
    arr = torch.tensor(
        [
            [6, 9, 12, 15.0, 18, 18, 18, 19, 17, 19, 18, 18],
            [34, 35, 36, 35, 35, 42.0, 49, 56, 55, 57, 56, 56],
            [23, 24, 24, 23, 22, 23, 23, 23, 26, 29, 29, 30],
        ]
    )
    print("arr.size() = ", arr.size())
    label = torch.tensor([1, 2, 3])
    feats = model.forward(arr, label=label)
    print(feats)
    print()
    print(model._accuracy.get_metric())
