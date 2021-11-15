from allennlp_series.model.model_archive.prior_models import *
from allennlp_series.model.model_archive.operations import *

import numpy as np
import torch.nn as nn
import torch
from allennlp_series.common.constants import *

from typing import Tuple, List


class ThreeLayerLayoutPredictionProgram(nn.Module):
    """
    create operator layer 1
    create operator layer 2
    create accumulator layer
    """

    #
    """
    fwd pass
    - inp: bs, length
    - normalize: bs, length
    - layer1: bs, channels, length
    - reshape: bs*channels, length
    - layer2: bs*channels, channels, length
    - reshape: bs, channels, channels, length
    - accumulator: bs, channels, channels, acc_feats
    - reshape: bs, channels*channels*acc_feats 
    """

    def __init__(
        self,
        inp_length=12,
        operations_conv_type: str = "all",
        operations_acc_type: str = "max",
        use_signum: bool = False,
        use_l1_loss: bool = True,
        prior_model_type: str = "prior",  # prior, newprior, simpleprior
        series_type: str = SERIES_TYPE_SINGLE,
        run_without_norm: bool = False,
        use_vertical_opers: bool = False,
        num_cols_in_multiseries: int = None,
        vert_operations_set_type: str = None,
        use_inference_network_debug: bool = None,
        use_test_time_argmax: bool = False,
        use_inference_network: bool = False,
    ):
        super().__init__()
        assert series_type in SERIES_TYPES_LIST
        self.series_type = series_type
        self.use_inference_network = use_inference_network
        self.run_without_norm = run_without_norm
        self.use_test_time_argmax = use_test_time_argmax
        if use_inference_network:
            assert not use_vertical_opers, "Not implemented yet"
        self.use_inference_network_debug = use_inference_network_debug

        self.layer1: ConvOperatorsChoice = ConvOperatorsChoice(
            inp_length=inp_length,
            operations_type=operations_conv_type,
            use_signum=use_signum,
        )
        self.layer1_numfeats = self.layer1.num_features
        self.num_operators_l1 = self.layer1.num_operators  # =1
        self.num_output_operators_l1 = self.layer1.out_channels
        self.layer2: ConvOperatorsChoice = ConvOperatorsChoice(
            inp_length=self.layer1_numfeats,
            operations_type=operations_conv_type,
            use_signum=use_signum,
        )

        self.layer2_numfeats = self.layer2.num_features
        self.num_operators_l2 = self.layer2.num_operators  # out_channels
        self.num_output_operators_l2 = self.layer2.out_channels  # =1

        if prior_model_type == "prior":
            self.layer3 = AccumulatorOperator(
                inp_length=self.layer2_numfeats, operations_type=operations_acc_type
            )
        else:
            self.layer3 = AccumulatorOperatorChoice(
                inp_length=self.layer2_numfeats, operations_type=operations_acc_type
            )

        self.use_l1_loss = use_l1_loss
        self.num_operators_l3 = self.layer3.num_operators  #
        self.num_output_operators_l3 = self.layer3.num_features  #
        self.prior_model_type = prior_model_type

        # Vertical Operator
        self.use_vertical_opers = use_vertical_opers
        if use_vertical_opers:
            assert series_type == SERIES_TYPE_MULTI
            self._vert_featurizer: VertTwoSeriesConvOperatorChoice = (
                VertTwoSeriesConvOperatorChoice(
                    inp_length=num_cols_in_multiseries,
                    operations_type=vert_operations_set_type,
                )
            )

        # Prior network
        if prior_model_type == "prior":
            assert (
                not use_vertical_opers
            ), "use_vertical_opers not implemented for 'prior' - try 'simpleprior' instead"
            self.number_of_programs = (
                self.num_operators_l1 * self.num_operators_l2
            )  # * self.layer3.num_features
            self.prior_network: PriorModel = PriorModel(
                series_length=inp_length, num_programs=self.number_of_programs
            )

            self.num_features = (
                self.num_output_operators_l1
                * self.num_output_operators_l1
                * self.layer3.num_features
                + self.prior_network.embedding_dim
            )
            # i,.e. 1*1*self.layer3.num_features = self.layer3.num_features
        elif prior_model_type == "simpleprior":
            self.number_of_programs = (
                self.num_operators_l1 * self.num_operators_l2 * self.num_operators_l3
            )
            num_layers = 3  # 4 if use_vertical_opers else None
            share_embeddings = (
                "-1_0_-1"  #'-1_-1_0_-1' if use_vertical_opers else '-1_0_-1'
            )
            vertical_num_programs = (
                self._vert_featurizer.num_operators if use_vertical_opers else None
            )
            self.prior_network = JoinedPriorModel(
                series_length=inp_length,
                num_layers=num_layers,
                num_programs=[
                    self.num_operators_l1,
                    self.num_operators_l2,
                    self.num_operators_l3,
                ],
                share_embeddings=share_embeddings,
                embedding_dim=5,
                use_vertical_opers=use_vertical_opers,
                num_cols_in_multiseries=num_cols_in_multiseries,
                vertical_num_programs=vertical_num_programs,
            )
            self.num_features = (
                self.num_output_operators_l1
                * self.num_output_operators_l2
                * self.num_output_operators_l3
                + num_layers * self.prior_network.embedding_dim
            )
            if use_vertical_opers:
                pass
                # self.num_features += self.prior_network.embedding_dim
                # self.num_features

        import copy

        if use_inference_network:
            self.posterior_model = JoinedPosteriorModel(
                series_length=inp_length,
                num_layers=3,
                num_programs=[
                    self.num_operators_l1,
                    self.num_operators_l2,
                    self.num_operators_l3,
                ],
                share_embeddings="-1_0_-1",
                label_embedding_dim=5,
                label_model_typ="avg_emb",
                prior_type="simple_posterior",
                embedding_dim=5,
                debug_mode=use_inference_network_debug,
                use_vertical_opers=use_vertical_opers,
            )

            # self.posterior_model.prior_network = copy.deepcopy(self.prior_network.prior_network) # TODO *********
            # print("======= use_inference_network: POSTERIOR")
            # print(list(self.posterior_model.named_parameters()))
            # print("======= use_inference_network: PRIOR")
            # print(list(self.prior_network.named_parameters()))

            if use_inference_network_debug:  ##****
                attrs_prior = vars(self.prior_network)
                attrs_posterior = vars(self.posterior_model)
                for attr, vals in attrs_prior.items():
                    print("PRIOR: ", attr, vals)
                    print("POSTERIOR: ", attr, attrs_posterior.get(attr, None))
                    print("=======")
                # assert False
                # self.posterior_model.prior_network = self.prior_network.prior_network
                # import copy
                self.posterior_model.prior_network = copy.deepcopy(
                    self.prior_network.prior_network
                )
                for attr, vals in attrs_prior.items():
                    print("re:PRIOR: ", attr, vals)
                    print("re:POSTERIOR: ", attr, attrs_posterior.get(attr, None))
                    print("=======")
                print("======= POSTERIOR")
                print(list(self.posterior_model.named_parameters()))
                print("======= PRIOR")
                print(list(self.prior_network.named_parameters()))
                # self.posterior_model.prior_network[1] = copy.deepcopy(self.prior_network.prior_network[1])

    def forward(self, series: torch.FloatTensor, label_text: torch.LongTensor = None):

        predictor_network = self.prior_network
        if self.use_inference_network:
            predictor_network = self.posterior_model
            # if self.use_inference_network_debug: # self.prior_network #self.posterior_model # **** TODO
            #     predictor_network = self.prior_network
            assert not self.use_vertical_opers, "Not implemented yet"

        program_selection_method = "sample"
        if self.use_test_time_argmax:
            if not self.training:
                program_selection_method = "factored_argmax"

        vert_sampled_program, vert_sampled_program_emb, vert_sampled_program_logprob = (
            None,
            None,
            None,
        )
        if self.use_vertical_opers:
            # series: bs, num_series, length
            bs, num_series, len_series = series.size()
            # --- note this being run on unnormalized ** TODO
            (
                vert_sampled_program,
                vert_sampled_program_emb,
                vert_sampled_program_logprob,
            ) = predictor_network.forward_vertical(series.view(bs, -1))
            more_cols = torch.cat(
                [
                    self._vert_featurizer.forward(
                        norm_series_bsi.unsqueeze(0), choice_num=sampled_program_i_bsi
                    )
                    for sampled_program_i_bsi, norm_series_bsi in zip(
                        vert_sampled_program, series
                    )
                ]
            )
            series = torch.cat([series, more_cols], dim=1)  # bs, n1+n2, length
            if self.use_test_time_argmax:
                raise NotImplementedError

        bs = series.size()[0]

        label_text_extended = label_text
        series_without_norm = series  # None
        norm_series = None

        if self.series_type == SERIES_TYPE_MULTI:

            series_without_norm = series.view(-1, series.size()[2])
            norm_series = normalize_with_maxval(series)
            norm_series_orig = np.array(norm_series)  # keep a copy
            series_row_cnt = series.size()[1]
            bs_old = bs
            bs = bs_old * series_row_cnt
            norm_series = norm_series.view(bs, -1)

            # label_text: bs, 1
            # change to bsnew, 1; by first changing to bs,1,1 then repeat to [1,rep,1), then bs*rep,1
            if self.use_inference_network:
                label_text_extended: torch.LongTensor = (
                    label_text.unsqueeze(1).repeat([1, series_row_cnt, 1]).view(bs, -1)
                )
                # print("label_text_extended: ", label_text_extended.size())

            # series_without_norm = series
            # norm_series_orig = normalize_with_maxval(series)  # copy. 1,length
            # norm_series_orig = norm_series
            # print("series = ", series)
            # print("norm_series = ", norm_series)

        else:
            norm_series = normalize_with_maxval(series)

        if self.run_without_norm:
            norm_series = series_without_norm

        # print("ThreeLayerLayoutPredictionProgram: norm_series = ", norm_series.size())
        sampled_program, sampled_program_emb, sampled_program_logprob = None, None, None
        sampled_program_i, sampled_program_j, sampled_program_k = None, None, None

        if self.prior_model_type == "prior":

            assert program_selection_method == "sample"
            if self.use_inference_network:
                # program_dist_vals = predictor_network.forward(series, label_text)  # bs, num_programs
                program_dist_vals = predictor_network.forward(
                    series, label_text_extended
                )
                # bs, num_programs
            else:
                program_dist_vals = predictor_network.forward(series)
                # bs, num_programs
            # *** notice this is being done on series and not norm_Series
            # program_dist_vals = predictor_network.forward(norm_series_orig)
            # TODO - need to shift to this instead of above. need to check any impact on results
            program_dist_logits = program_dist_vals["logits"]
            (
                sampled_program,
                sampled_program_emb,
                sampled_program_logprob,
            ) = predictor_network.sample(logits=program_dist_logits)
            sampled_program_ij: Tuple[List, List] = predictor_network.get_layer_wise(
                sampled_program, num_operations_per_layer=self.num_operators_l1
            )
            sampled_program_i, sampled_program_j = sampled_program_ij
            sampled_program_k = None

        elif self.prior_model_type == "simpleprior":

            # sampled_program, sampled_program_emb, sampled_program_logprob = predictor_network.forward(norm_series)
            # sampled_program, sampled_program_emb, sampled_program_logprob = predictor_network.forward(series_without_norm)

            if self.use_inference_network:
                (
                    sampled_program,
                    sampled_program_emb,
                    sampled_program_logprob,
                ) = predictor_network.forward(
                    series_without_norm,
                    label_text_extended,
                    program_selection_method=program_selection_method,
                )
            else:
                (
                    sampled_program,
                    sampled_program_emb,
                    sampled_program_logprob,
                ) = predictor_network.forward(
                    series_without_norm,
                    program_selection_method=program_selection_method,
                )

            # TODO - need to change this to norm_series. old results of simple prior are with series
            # --- need to add label text to the above call in case of inference network

            sampled_program_i, sampled_program_j, sampled_program_k = (
                sampled_program[0],
                sampled_program[1],
                sampled_program[2],
            )

            # TODO - in case of inference network, use gumbel softmax instead of reinforce -- keep option for both
            #  -- for now using reinforce
            # Also when using Gumbel Softmax: reward won't be used

        # now run layer1 and layer 2s to get features
        # append the sampled_program_emb to the features
        # also keep track of log_prob of the distribution
        # in the type-classifier code, the loss from the clf would be the reward
        # so this code should return the log_probs, samples. and that would be combined with rewards
        # to get the additionak loss term

        # print("sampled_program_i = ", sampled_program_i)
        # print("norm_series: ", norm_series.size())
        # print("self.layer1.forward(norm_series[0], choice_num=sampled_program_i_bsi): ",
        #       self.layer1.forward(norm_series[0:1], choice_num=sampled_program_i[0]) )

        out = torch.cat(
            [
                self.layer1.forward(
                    norm_series_bsi.unsqueeze(0), choice_num=sampled_program_i_bsi
                )
                for sampled_program_i_bsi, norm_series_bsi in zip(
                    sampled_program_i, norm_series
                )
            ]
        )
        # out_re = out.view(bs * self.num_output_operators_l1, -1)  # bs*num, inp-length-1
        out_re = out.view(bs * 1, -1)  # bs*1, inp-length-1

        out2 = torch.cat(
            [
                self.layer2.forward(
                    out_re_bsj.unsqueeze(0), choice_num=sampled_program_j_bsj
                )
                for sampled_program_j_bsj, out_re_bsj in zip(sampled_program_j, out_re)
            ]
        )
        # out2 = out2.view(bs, self.num_output_operators_l1, self.num_output_operators_l2, -1)  # bs, num, num, featsize
        out2 = out2.view(bs * 1, -1)  # bs* num=1 * num=1, featsizes
        # print(" ** out2: ", out2.size())

        if sampled_program_k is None:
            out3 = self.layer3(out2)
        else:
            out3 = torch.cat(
                [
                    self.layer3.forward(out2_bsk, choice_num=sampled_program_k_bsk)
                    for sampled_program_k_bsk, out2_bsk in zip(sampled_program_k, out2)
                ]
            )
        out3 = out3.view(bs, -1)

        if self.series_type == SERIES_TYPE_MULTI:
            out3 = out3.view(bs_old, -1)
            sampled_program_emb = sampled_program_emb.view(bs_old, -1)

        # print("sampled_program_emb : ", sampled_program_emb.size())
        # print("out3 : ", out3.size())

        if self.use_vertical_opers:
            # vert_sampled_program, vert_sampled_program_emb, vert_sampled_program_logprob
            # print("sampled_program_logprob = ", sampled_program_logprob.size())
            # print("vert_sampled_program_logprob = ", vert_sampled_program_logprob.size())
            sampled_program_logprob = sampled_program_logprob.view(
                bs_old, -1
            ) + vert_sampled_program_logprob.view(-1, 1)
            sampled_program_logprob = sampled_program_logprob.view(-1)
            # print("sampled_program: ", len(sampled_program), sampled_program[0].size(), "vert_sampled_program: ", vert_sampled_program.size())
            sampled_program = sampled_program + [vert_sampled_program]
            # print("sampled_program_emb: ",  sampled_program_emb.size(), "vert_sampled_program_emb: ", vert_sampled_program_emb.size())
            sampled_program_emb = torch.cat(
                [sampled_program_emb, vert_sampled_program_emb], dim=1
            )
            # print("sampled_program_emb: ",  sampled_program_emb.size())

        ret = {
            "computed_program_output": out3,
            "sampled_program_emb": sampled_program_emb,
            "log_prob": sampled_program_logprob,
            "action": sampled_program,
        }

        if self.use_inference_network:
            kl_loss = self._compute_kl(norm_series, label_text_extended)
            ret["kl_loss"] = kl_loss

        return ret

    def _compute_kl(
        self, series: torch.FloatTensor, label_text: torch.LongTensor, k: int = 10
    ) -> torch.FloatTensor:
        """
        :param series:
        :param label_text:
        :return:
        - Construct the prior
        - Repeatedly sampled from the posterior [z1,z2,z3]
            - Compute posterior: log q(z)
            - Compute prior: log p(z)
            - Compute log q(z) - log p(z)
        - return the mean value ?
        """
        prior_model: JoinedPriorModel = self.prior_network
        posterior_model: JoinedPosteriorModel = self.posterior_model
        ret = 0.0
        for j in range(k):
            (
                sampled_program,
                sampled_program_emb,
                sampled_program_logprob,
            ) = posterior_model.forward(
                series, label_text, program_selection_method="sample"
            )
            logq = sampled_program_logprob
            logp = prior_model.get_prob_of_sample(series, sampled_program)
            ret += logq - logp
        # ret: is a tensor of size batch_size
        return ret / k

    def _compute_exact_kl(
        self, series: torch.FloatTensor, label_text: torch.LongTensor
    ) -> torch.FloatTensor:
        """
        :param series:
        :param label_text:
        :return:
        - Construct the prior
        - Sum over all [z1,z2,z3]
            - Compute posterior: log q(z)
            - Compute prior: log p(z)
            - Compute log q(z) - log p(z)
        - return the mean value ?
        """
        prior_model: JoinedPriorModel = self.prior_network
        posterior_model: JoinedPosteriorModel = self.posterior_model
        program_space = [
            posterior_model.prior_network[j].num_programs
            for j in range(len(posterior_model.prior_network))
        ]
        program_space_cnt = 1
        for pp in program_space:
            program_space_cnt *= pp
        ret = 0.0
        for j in range(program_space_cnt):
            sampled_program = self._get_each_layer(j, program_space)
            logp = prior_model.get_prob_of_sample(series, sampled_program)
            logq = posterior_model.get_prob_of_sample(
                series, sampled_program
            )  # TODO need to implement this
            ret += logq * (logq - logp)
        # ret: is a tensor of size batch_size
        return ret / program_space_cnt

    def _get_each_layer(self, cnt, program_space):
        tmp = cnt
        ret = []
        for pp in program_space:
            m = tmp % pp
            ret = [m] + ret
            tmp = tmp // pp
        return ret


if __name__ == "__main__":

    model = ThreeLayerLayoutPredictionProgram(
        operations_conv_type="configa", operations_acc_type="max_min_mean"
    )
    arr = torch.tensor(
        [
            [
                0.0100,
                0.0333,
                -0.0266,
                -0.0166,
                0.0333,
                0.0133,
                -0.0300,
                0.0033,
                0.0100,
                0.0333,
                -0.0266,
                -0.0166,
            ],
            [
                0.3100,
                0.0333,
                -0.0266,
                -0.0166,
                0.0333,
                0.0133,
                -0.0300,
                0.3033,
                0.0100,
                0.0333,
                -0.0266,
                -0.0166,
            ],
        ]
    )
    print("arr.size() = ", arr.size())
    feats = model(arr)
    print(feats)
