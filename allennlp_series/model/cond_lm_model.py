from typing import Dict, List, Tuple, Union, Any
import torch
import numpy as np
import os
import logging
from torch.nn.modules.rnn import LSTMCell
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn.util import get_text_field_mask
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import Perplexity
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util

from allennlp_series.common.constants import *
from allennlp_series.training.metrics import CocovalsMeasures
from allennlp_series.training.metrics.diversity_evals import DiversityEvals
from allennlp_series.training.metrics.program_activation_analysis import (
    ProgramActivationEvals,
)
import allennlp_series.model.utils as utils


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
logger = logging.getLogger(__name__)

#
# class _SoftmaxLoss(torch.nn.Module):
#     """
#     Given some embeddings and some targets, applies a linear layer
#     to create logits over possible words and then returns the
#     negative log likelihood.
#     """
#
#     def __init__(self, num_words: int, embedding_dim: int) -> None:
#         super().__init__()
#
#         self.tie_embeddings = False
#
#         self.softmax_w = torch.nn.Parameter(
#             torch.randn(embedding_dim, num_words) / np.sqrt(embedding_dim)
#         )
#         self.softmax_b = torch.nn.Parameter(torch.zeros(num_words))
#
#     def forward(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#
#         # embeddings is size (n, embedding_dim)
#         # targets is (batch_size, ) with the correct class id
#         # Does not do any count normalization / divide by batch size
#         probs = torch.nn.functional.log_softmax(
#             torch.matmul(embeddings, self.softmax_w) + self.softmax_b, dim=-1
#         )
#
#         return torch.nn.functional.nll_loss(probs, targets.long(), reduction="sum")


@Model.register("cond_language_model")
class ConditionalLanguageModel(Model):
    """
    The `LanguageModel` applies a "contextualizing"
    `Seq2SeqEncoder` to uncontextualized embeddings, using a `SoftmaxLoss`
    module (defined above) to compute the language modeling loss.

    If bidirectional is True,  the language model is trained to predict the next and
    previous tokens for each token in the input. In this case, the contextualizer must
    be bidirectional. If bidirectional is False, the language model is trained to only
    predict the next token for each token in the input; the contextualizer should also
    be unidirectional.

    If your language model is bidirectional, it is IMPORTANT that your bidirectional
    `Seq2SeqEncoder` contextualizer does not do any "peeking ahead". That is, for its
    forward direction it should only consider embeddings at previous timesteps, and for
    its backward direction only embeddings at subsequent timesteps. Similarly, if your
    language model is unidirectional, the unidirectional contextualizer should only
    consider embeddings at previous timesteps. If this condition is not met, your
    language model is cheating.

    # Parameters

    vocab : `Vocabulary`
    text_field_embedder : `TextFieldEmbedder`
        Used to embed the indexed tokens we get in `forward`.
    contextualizer : `Seq2SeqEncoder`
        Used to "contextualize" the embeddings. As described above,
        this encoder must not cheat by peeking ahead.
    dropout : `float`, optional (default: None)
        If specified, dropout is applied to the contextualized embeddings before computation of
        the softmax. The contextualized embeddings themselves are returned without dropout.
    num_samples : `int`, optional (default: None)
        If provided, the model will use `SampledSoftmaxLoss`
        with the specified number of samples. Otherwise, it will use
        the full `_SoftmaxLoss` defined above.
    sparse_embeddings : `bool`, optional (default: False)
        Passed on to `SampledSoftmaxLoss` if True.
    bidirectional : `bool`, optional (default: False)
        Train a bidirectional language model, where the contextualizer
        is used to predict the next and previous token for each input token.
        This must match the bidirectionality of the contextualizer.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        program_emb_size: int = 5,
        hidden_dim: int = 5,
        embedding_dim: int = 5,
        dropout: float = None,
        target_namespace: str = "tokens",
        initializer: InitializerApplicator = None,
        eval_sanity_check_mode: bool = False,
        use_activation_evals: bool = False,
        model_programs=None,
        max_decoding_steps_generate: int = 10,
        model_name: str = None,
        use_bertscore_evals: bool = False,
        use_bow_decoder: bool = False,
        decoding_method: str = "greedy",
        sampling_top_p: float = 0.9,
        sampling_top_k: int = None,
        add_prog_emb_to_inp: bool = False,
        **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)

        self.eval_sanity_check_mode = eval_sanity_check_mode
        self._target_namespace = target_namespace
        self._program_emb_size = program_emb_size
        self._target_embedding_dim = embedding_dim  # program_emb_size #5
        self._decoder_output_dim = hidden_dim  # program_emb_size #5
        # self._softmax_loss = _SoftmaxLoss(
        #     num_words=vocab.get_vocab_size(), embedding_dim=self._target_embedding_dim
        # )  # not used **
        self.decoding_method = decoding_method
        self.sampling_top_p = sampling_top_p

        # This buffer is now unused and exists only for backwards compatibility reasons.
        self.register_buffer("_last_average_loss", torch.zeros(1))

        self._perplexity = Perplexity()
        self._ngram_overlap_eval = CocovalsMeasures(
            sanity_check_mode=eval_sanity_check_mode,
            compute_bert_score=use_bertscore_evals,
        )
        self._diversity_eval = DiversityEvals(model_name=model_name)
        self.use_activation_evals = use_activation_evals
        if use_activation_evals:
            self._program_activation_evals = ProgramActivationEvals(
                programs=model_programs
            )

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

        self._start_index = self.vocab.get_token_index(
            START_SYMBOL, self._target_namespace
        )
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)

        num_classes = self.vocab.get_vocab_size(self._target_namespace)
        self._target_embedder = Embedding(num_classes, self._target_embedding_dim)

        self._add_prog_emb_to_inp = add_prog_emb_to_inp
        self._decoder_input_dim = self._target_embedding_dim
        self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)

        # self._output_projection_layer = Linear(self._decoder_output_dim, num_classes)
        # self._output_projection_layer = Linear(self._decoder_output_dim+program_emb_size, num_classes)
        self._output_projection_layer = Linear(self._decoder_output_dim * 2, num_classes)
        # self._output_projection_layer =  self._target_embedder.weight
        # self._output_projection_layer_bias =  torch.nn.Parameter(torch.zeros(num_classes))

        self._program_to_hidden_projection = Linear(
            program_emb_size, self._decoder_output_dim
        )
        if self._add_prog_emb_to_inp:
            self._program_to_inp_projection = Linear(
                program_emb_size, self._target_embedding_dim
            )
        # self._program_to_output_projection = Linear(program_emb_size, num_classes)

        self._max_decoding_steps = max_decoding_steps_generate
        self.use_bow_decoder = use_bow_decoder
        if use_bow_decoder:
            self.bow_decoder_matrix = Linear(program_emb_size, num_classes)

        ##
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        num_params = sum([np.prod(p.size()) for p in model_parameters])
        print("[COND-LM] num params language model = ", num_params)
        ##

        if initializer is not None:
            initializer(self)

    def process_batch(
        self,
        program_embedding=None,
        num_decoding_steps=None,
        targets=None,
        generate_or_ppl="ppl",
        decoding_method: str = "greedy",
        sampling_top_p: float = 0.9,
        sampling_top_k: int = None,
    ):

        use_gold_targets = False
        # print("num_decoding_steps = ", num_decoding_steps)
        if self.training:
            assert generate_or_ppl == "ppl"
            use_gold_targets = True
        else:
            if generate_or_ppl == "ppl":
                use_gold_targets = True
            else:
                use_gold_targets = False

        last_predictions = None

        if False:  # True:
            batch_size = program_embedding.size()[0]
            decoder_hidden = self._program_to_hidden_projection(
                program_embedding
            )  # bs, h
        else:  # lesser params to train
            batch_size = targets.size()[0]
            decoder_hidden = torch.zeros(batch_size, self._decoder_output_dim)

        decoder_context = torch.zeros(batch_size, self._decoder_output_dim)
        if torch.cuda.is_available():
            decoder_hidden = decoder_hidden.cuda()
            decoder_context = decoder_context.cuda()

        step_logits = []
        step_probabilities = []
        step_predictions = []

        for timestep in range(num_decoding_steps):

            if use_gold_targets:
                input_choices = targets[:, timestep]
            else:
                if timestep == 0:
                    input_choices = targets[:, timestep]  # init with start symbols
                else:
                    input_choices = last_predictions

            decoder_input = self._prepare_decode_step_input(
                input_choices, decoder_hidden, program_embedding
            )
            decoder_hidden, decoder_context = self._decoder_cell(
                decoder_input, (decoder_hidden, decoder_context)
            )
            # (batch_size, num_classes)
            if len(program_embedding.size()) == 1:
                program_embedding = program_embedding.view(1, -1)
            program_embedding_proj = self._program_to_hidden_projection(
                program_embedding
            )
            output_projections = self._output_projection_layer(
                torch.cat([decoder_hidden, program_embedding_proj], dim=1)
            )
            # output_projections = F.linear(input=decoder_hidden,
            #                               weight=self._output_projection_layer,
            #                               bias=self._output_projection_layer_bias)
            # program_proj = self._program_to_output_projection(program_embedding)
            # output_projections = output_projections + program_proj
            if self.use_bow_decoder:
                output_projections = self.bow_decoder_matrix(program_embedding)

            step_logits.append(output_projections.unsqueeze(1))  # bs,1,vocab
            class_probabilities = F.softmax(output_projections, dim=-1)
            step_probabilities.append(class_probabilities.unsqueeze(1))

            if self.use_bow_decoder:
                _, predicted_classes = torch.kthvalue(
                    -class_probabilities, k=timestep + 1, dim=1
                )  ## generation through argmax
                # does assume that number of steps is smaller than the vocab
                # print("predicted_classes = ", predicted_classes)
            else:
                if decoding_method == "greedy":
                    _, predicted_classes = torch.max(
                        class_probabilities, 1
                    )  ## generation through argmax
                elif decoding_method == "sample":
                    # _, predicted_classes = torch.max(class_probabilities, 1)
                    # print(" *** predicted_classes : ", predicted_classes.size())
                    temperature = 1.0
                    if sampling_top_k is None:
                        top_k = 0
                    else:
                        top_k = sampling_top_k
                    top_p = sampling_top_p  # 0.9
                    bs = output_projections.size()[0]
                    assert bs == 1, "found bs = " + str(bs) + " , but wanted bs = 1"
                    # print("output_projections : ", output_projections.size())
                    logits = output_projections.view(-1) / temperature
                    filtered_logits = utils.top_k_top_p_filtering(
                        logits, top_k=top_k, top_p=top_p
                    )
                    probabilities = F.softmax(filtered_logits, dim=-1)
                    predicted_classes = torch.multinomial(probabilities, 1)
                    # print("predicted_classes : ", predicted_classes.size())
                else:
                    raise NotImplementedError
            last_predictions = predicted_classes
            # (batch_size, 1)
            step_predictions.append(last_predictions.unsqueeze(1))
            # for self.use_bow_decoder mode,
            # for now picking the topk words
            # note that due to output_projections = self.bow_decoder_matrix(program_embedding)
            # projections are same at every step

        return {
            "step_logits": step_logits,
            "step_probabilities": step_probabilities,
            "step_predictions": step_predictions,
        }

    def forward(  # type: ignore
        self,
        program_embedding: torch.FloatTensor,
        target_tokens: [str, torch.LongTensor],
        metadata: List[Dict[str, Any]] = None,
        mode="ppl",
        selected_program_id: List[int] = None,
    ) -> Dict[str, torch.Tensor]:

        # print("[CondLM] program_embedding = ", program_embedding)
        targets = None
        if len(program_embedding.size()) == 1:
            program_embedding = program_embedding.view(1, -1)
        batch_size = program_embedding.size()[0]
        # print("[CondLM] : target_tokens = ", target_tokens)
        assert mode in ["ppl", "generate"]

        if target_tokens:

            targets = target_tokens["tokens"]
            target_sequence_length = targets.size()[1]
            # The last input from the target is either padding or the end symbol. Either way, we
            # don't have to process it.
            num_decoding_steps = (
                target_sequence_length - 1
            )  # --> start will suupply here explicitly

            # targets: bs, timesteps
            # shape (batch_size, timesteps, embedding_size)
            # embeddings = self._target_embedder.forward(targets)

        else:

            num_decoding_steps = self._max_decoding_steps

        # print("program_embedding: ", program_embedding)
        # print("targets: ", type(targets))
        vals_loss = self.process_batch(
            program_embedding, num_decoding_steps, targets, generate_or_ppl=mode
        )
        step_logits, step_probabilities, step_predictions = (
            vals_loss["step_logits"],
            vals_loss["step_probabilities"],
            vals_loss["step_predictions"],
        )

        # step_logits is a list containing tensors of shape (batch_size, 1, num_classes)
        # This is (batch_size, num_decoding_steps, num_classes)

        logits = torch.cat(step_logits, 1)
        class_probabilities = torch.cat(step_probabilities, 1)
        all_predictions = torch.cat(step_predictions, 1)

        output_dict = {
            "logits": logits,
            "class_probabilities": class_probabilities,
            "predictions": all_predictions,
        }

        if target_tokens:

            target_mask = get_text_field_mask(target_tokens)
            # loss = self._get_loss(logits, targets, target_mask)
            loss = self._get_loss(logits, targets, target_mask, average=None)
            output_dict["logprob_ylabel_given_z"] = -loss
            average_loss = torch.mean(loss)  # loss #.data.cpu().numpy()
            # if self.unconditional_lm:
            #     output_dict["loss"] = average_loss  # torch.mean(loss)

            if not self.training:
                if mode == "generate":
                    all_generated = []
                    all_target = []
                    i2v = self.vocab.get_index_to_token_vocabulary("tokens")
                    num_decoding_steps = self._max_decoding_steps  # ** new
                    vals_sample = self.process_batch(
                        program_embedding,
                        num_decoding_steps,
                        targets,
                        generate_or_ppl="generate",
                        decoding_method=self.decoding_method,
                        sampling_top_p=self.sampling_top_p,
                    )
                    step_predictions = vals_sample["step_predictions"]
                    # step_predictions: time_steps,batch_size,1
                    for b in range(batch_size):
                        # print("batch_size = ", batch_size)
                        step_predictions_b = [
                            pred_i[b].data.item() for pred_i in step_predictions
                        ]
                        step_predictions_b = [p for p in step_predictions_b if p != 0]
                        if self.use_bow_decoder:
                            pass
                        else:
                            end_idx = (
                                step_predictions_b.index(self._end_index)
                                if self._end_index in step_predictions_b
                                else len(step_predictions_b)
                            )
                            if end_idx != -1:
                                step_predictions_b = step_predictions_b[:end_idx]
                        predicted_str = " ".join([i2v[pi] for pi in step_predictions_b])
                        targets_b = [pred_i.data.item() for pred_i in targets[b]]
                        targets_b = [p for p in targets_b if p != 0]
                        targets_b = targets_b[1:-1]  # ** removing start and end index
                        target_str = " ".join([i2v[pi] for pi in targets_b])
                        # logger.debug(f" ************** [generate] target_str = {target_str}")
                        # logger.debug(f" ************** [generate] predicted_str = {predicted_str}")
                        print(" ************** [generate] target_str =",  target_str)
                        print(" ************** [generate] predicted_str = ", predicted_str)
                        all_generated.append(predicted_str)
                        all_target.append(target_str)
                        id = metadata[b]["idx"]
                        self._ngram_overlap_eval(predicted_str, target_str, id)
                        self._diversity_eval(predicted_str, typ="generated")
                        self._diversity_eval(target_str, typ="gt")
                        if self.use_activation_evals:
                            self._program_activation_evals(
                                predicted_str,
                                typ="generated",
                                program_id=selected_program_id[b],
                            )
                            self._program_activation_evals(
                                target_str, typ="gt", program_id=selected_program_id[b]
                            )
                    output_dict.update(
                        {
                            "generate_all_generated": all_generated,
                            "generate_all_target": all_target,
                        }
                    )
            else:
                if mode == "generate":
                    raise NotImplementedError(
                        "generate mode not implemented for training mode"
                    )

        return output_dict

    def log_ppl(self, avg_loss):
        self._perplexity(avg_loss)

    def get_metrics(self, reset: bool = False):
        ret = {"perplexity": self._perplexity.get_metric(reset=reset)}
        if (not self.training) and reset: # and (not self.unconditional_lm):
            ret.update(self._ngram_overlap_eval.get_metric(reset))
            ret.update(self._diversity_eval.get_metric(reset))
            if self.use_activation_evals:
                ret.update(self._program_activation_evals.get_metric(reset))
        return ret

    def _prepare_decode_step_input(
        self,
        input_indices: torch.LongTensor,
        decoder_hidden_state: torch.LongTensor = None,
        program_embedding: torch.FloatTensor = None,
    ) -> torch.Tensor:

        embedded_input = self._target_embedder.forward(input_indices)
        # if not self.unconditional_lm:
        if self._add_prog_emb_to_inp:
            # program_emb_extended = program_embedding
            program_emb_extended = self._program_to_inp_projection(
                program_embedding
            )
            embedded_input = torch.cat([embedded_input, program_emb_extended], -1)
            # print("embedded_input : ", embedded_input.size())
            # (batch_size, encoder_output_dim + target_embedding_dim)
        return embedded_input

    @staticmethod
    def _get_loss(
        logits: torch.FloatTensor,
        targets: torch.LongTensor,
        target_mask: torch.LongTensor,
        average: str = "batch",
    ) -> torch.FloatTensor:
        """
        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.
        The length of ``targets`` is expected to be greater than that of ``logits`` because the
        decoder does not need to compute the output corresponding to the last timestep of
        ``targets``. This method aligns the inputs appropriately to compute the loss.
        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        # relevant_targets = targets.contiguous()  # (batch_size, num_decoding_steps)
        # relevant_mask = target_mask.contiguous()  # (batch_size, num_decoding_steps)
        relevant_targets = targets[
            :, 1:
        ].contiguous()  # (batch_size, num_decoding_steps)
        relevant_mask = target_mask[
            :, 1:
        ].contiguous()  # (batch_size, num_decoding_steps)
        loss = util.sequence_cross_entropy_with_logits(
            logits, relevant_targets, relevant_mask, average=average
        )
        # print('_get loss : ', loss)
        return loss
