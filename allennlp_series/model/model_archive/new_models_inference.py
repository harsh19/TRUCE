# import torch.nn as nn
#
# from allennlp.nn.initializers import InitializerApplicator
# from allennlp.data.vocabulary import Vocabulary
# from allennlp.models.model import Model
# from allennlp.training.metrics.average import Average
#
# from allennlp_series.model.new_prior_models import *
# from allennlp_series.training.metrics.confusion_matrix import ConfusionMatrix, F1MeasureCustom
# from allennlp_series.training.metrics.distractor_evals import DistractorEvals
# from allennlp_series.model.cond_lm_model import ConditionalLanguageModel
# from allennlp_series.model.new_modules import *
# from allennlp_series.model.trainable_modules import TrainableLocateModule, TrainableAttendModule
# import math
#
#
# # ******************** ******************** # TODO
# TrainableTwoLayerAttendModule=TrainableAttendModule
# # ******************** ********************
#
# def create_all_instances(depth: int = 1,
#                          use_combine_new_defn: bool = False,
#                          program_set='type1',
#                          trainable_locate: bool = False,
#                          trainable_attend: bool = False,
#                          init_trainable_with_hardcoded_vals : bool = False):
#     assert depth == 1
#     instances = {}
#
#     instances['attend'] = []
#     init_attend = init_trainable_with_hardcoded_vals
#     if trainable_attend:
#         fixed_module = False
#     else:
#         fixed_module = True
#         init_attend = True
#     instances['attend'].append(TrainableTwoLayerAttendModule(operator_name='a1',
#                                      operator_init_type='increase' if init_attend else 'none',
#                                      fixed_module=fixed_module))
#     instances['attend'].append(TrainableTwoLayerAttendModule( operator_name='a2',
#                                     operator_init_type='decrease' if init_attend else 'none',
#                                     fixed_module=fixed_module))
#     if program_set in ['type2']:
#         instances['attend'].append(TrainableTwoLayerAttendModule(operator_name='a3',
#                                      operator_init_type='peak' if init_attend else 'none',
#                                      fixed_module=fixed_module))
#         instances['attend'].append(TrainableTwoLayerAttendModule(operator_name='a4',
#                                     operator_init_type='trough' if init_attend else 'none',
#                                     fixed_module=fixed_module))
#
#     instances['locate'] = []
#     sz = 12 #+ 2
#     if trainable_attend:
#         # sz=9+2
#         sz = 10+2 # single layer trainable
#     init_locate = init_trainable_with_hardcoded_vals
#     if trainable_locate:
#         fixed_module = False
#     else:
#         fixed_module = True
#         init_locate = True
#     instances['locate'].append(
#         TrainableLocateModule(operator_type='begin' if init_locate else None,
#                               operator_name='l1',
#                               inp_length=sz,
#                               fixed_module=fixed_module))
#     instances['locate'].append(
#         TrainableLocateModule(operator_type='middle' if init_locate else None,
#                               operator_name='l2',
#                               inp_length=sz,
#                               fixed_module=fixed_module))
#     instances['locate'].append(
#         TrainableLocateModule(operator_type='end' if init_locate else None,
#                               operator_name='l3',
#                               inp_length=sz,
#                               fixed_module=fixed_module))
#     if False:  #True: # Used for set2_learnboth_initneither_exp3. TODO - parameterize number of additional modules
#         assert not init_locate
#         instances['locate'].append(
#             TrainableLocateModule(operator_type=None,
#                                   operator_name='l4',
#                                   inp_length=sz,
#                                   fixed_module=fixed_module))
#         instances['locate'].append(
#             TrainableLocateModule(operator_type=None,
#                                   operator_name='l5',
#                                   inp_length=sz,
#                                   fixed_module=fixed_module))
#         instances['locate'].append(
#             TrainableLocateModule(operator_type=None,
#                                   operator_name='l6',
#                                   inp_length=sz,
#                                   fixed_module=fixed_module))
#
#     instances['combine'] = [CombineModule(operator_type='combine_exists',
#                                           operator_name='combine_exists',
#                                           use_new_defn=use_combine_new_defn)]
#     return instances
#
#
# model_type_to_testtimedefault = {
#     'marginalize': 'marginalize',
#     'marginalize_new': 'marginalize_new',
#     'prior_reinforce': 'sample'
# }
#
#
# def get_text_field_mask(text_field_tensors,
#                         num_wrapping_dims: int = 0) -> torch.LongTensor:
#     # if "mask" in text_field_tensors:
#     # print("text_field_tensors = ", text_field_tensors)
#     return text_field_tensors['tokens'] != 0
#     # return text_field_tensors["mask"]
#
#
# @Model.register("new_clf")
# class NewThreeLayerLayoutPredictionProgram(Model):
#     '''
#     create operator layer 1
#     create operator layer 2
#     create accumulator layer
#     '''
#     #
#     '''
#     fwd pass
#     - inp: bs, length
#     - normalize: bs, length
#     - layer1: bs, channels, length
#     - reshape: bs*channels, length
#     - layer2: bs*channels, channels, length
#     - reshape: bs, channels, channels, length
#     - accumulator: bs, channels, channels, acc_feats
#     - reshape: bs, channels*channels*acc_feats
#     '''
#
#     def __init__(self,
#                  vocab: Vocabulary,
#                  inp_length=12,
#                  operations_conv_type: str = 'all',
#                  operations_acc_type: str = 'max',
#                  series_type: str = SERIES_TYPE_SINGLE,
#                  use_inference_network_debug: bool = None,
#                  test_time_method_type: str = None,
#                  clf_type: str = 'linear',
#                  embedding_dim: int = 6,
#                  num_labels: int = 5,
#                  negative_class_wt: float = 1.0,
#                  prior_loss_wt: float = 1.0,
#                  clf_loss_wt: float = 1.0,
#                  use_inference_network: bool = False,
#                  model_type: str = 'marginalize',
#                  reinforce_baseline: str = None,  # None, 'mean_std'
#                  reinforce_num_samples: int = 3,
#                  init_program_emb_with_onehot: bool = False,
#                  use_combine_new_defn: bool = False,
#                  use_prior_potential: bool = False,
#                  train_unary_prior: bool = None,
#                  program_set_type: str = 'type1',
#                  program_trainable_locate: bool = False,
#                  program_trainable_attend: bool = False,
#                  text_hidden_dim: int = 5,
#                  text_embedding_dim: int = 5,
#                  task_setup_type: str = 'classification',  # classification, text
#                  text_eval_sanity_check_mode: bool = False,
#                  use_distractors_evals: bool = False,
#                  model_dict_load_file_path: str = None,
#                  init_trainable_with_hardcoded_vals: bool = False,
#                  initializer: InitializerApplicator = InitializerApplicator()):
#
#         super().__init__(vocab)
#         assert series_type in SERIES_TYPES_LIST
#         self.series_type = series_type
#         self.use_inference_network = use_inference_network
#         if test_time_method_type is None:
#             test_time_method_type = model_type_to_testtimedefault[model_type]
#         assert test_time_method_type in ['sample', 'argmax', 'marginalize', 'marginalize_new']
#         self.test_time_method_type = test_time_method_type
#         self.num_labels = num_labels
#         self.negative_class_wt = negative_class_wt
#         self.model_type = model_type
#         self.reinforce_baseline = reinforce_baseline
#         self.task_setup_type = task_setup_type
#         if negative_class_wt != 1.0 and num_labels > 2:  # not yet implemented
#             raise NotImplementedError
#         self.program_set_type = program_set_type
#         assert program_set_type in ['type1', 'type2']
#         self.instances = instances = create_all_instances(use_combine_new_defn=use_combine_new_defn,
#                                          program_set=program_set_type,
#                                          trainable_locate=program_trainable_locate,
#                                          trainable_attend=program_trainable_attend,
#                                          init_trainable_with_hardcoded_vals=init_trainable_with_hardcoded_vals)
#         assert model_type in ['marginalize', 'marginalize_new', 'prior_reinforce', 'inference_nw']
#         self.clf_loss_wt = clf_loss_wt
#         self.reinforce_num_samples = reinforce_num_samples
#         self.use_prior_potential = use_prior_potential
#         if use_prior_potential:
#             assert model_type == 'marginalize_new'
#         if train_unary_prior is None:
#             train_unary_prior = False
#             if use_prior_potential:
#                 train_unary_prior = True
#         self.train_unary_prior = train_unary_prior
#         self.text_eval_sanity_check_mode = text_eval_sanity_check_mode
#         if text_eval_sanity_check_mode:
#             assert self.task_setup_type == 'text'
#         self.program_trainable_locate = program_trainable_locate
#         self.program_trainable_attend = program_trainable_attend
#
#         self.programs = []
#         self.num_programs = 0
#         dct = {}
#         for i in range(len(instances['locate'])):
#             dct['locate'] = i
#             for j in range(len(instances['attend'])):
#                 dct['attend'] = j
#                 for k in range(len(instances['combine'])):
#                     dct['combine'] = k
#                     programi = SimpleProgramType(dct, instances)
#                     # add module
#                     self.num_programs += 1
#                     self.add_module('program' + str(self.num_programs), programi)
#                     self.programs.append(programi)
#                     print("[NewThreeLayerLayoutPredictionProgram]: ", self.num_programs - 1, programi.__str__)
#
#         self.prior_model = EnumerateAllPrior(num_programs=self.num_programs,
#                                              embedding_dim=embedding_dim,
#                                              init_program_emb_with_onehot=init_program_emb_with_onehot)
#
#         if self.task_setup_type == 'classification':
#             in_features = embedding_dim
#             out_features = num_labels
#             if clf_type == 'linear':
#                 h = in_features
#                 layer2 = torch.nn.Linear(h, out_features)
#                 self._classification_layer = nn.Sequential(layer2)
#                 initializer(self._classification_layer)
#
#                 # layer2.weight: ** out_features, h
#                 # for j in  range(1,out_features):
#                 #     layer2.weight.data[j,:] = layer2.weight.data[0,:]
#                 # for i in  range(1,num_labels):
#                 #     layer2.weight.data[:,i] = layer2.weight.data[:,0]
#                 # layer2.weight.bias =
#
#                 with torch.no_grad():
#                     layer2.weight.data.fill_(0)
#                     layer2.bias.data.fill_(0)
#
#                 # --> maybe simply init to all zeros ?
#                 print("self._classification_layer[0] ::: ", self._classification_layer[0].weight)
#
#             elif clf_type == 'two_layer':
#                 raise NotImplementedError  # todo proper init like for linear
#                 # h = int(in_features // 4)
#                 h = int(in_features // 2)
#                 layer1 = torch.nn.Linear(in_features, h)
#                 layer2 = torch.nn.Linear(h, out_features)
#                 self._classification_layer = nn.Sequential(
#                     layer1,
#                     nn.Sigmoid(),
#                     layer2)
#                 initializer(self._classification_layer)
#             elif clf_type == 'three_layer':
#                 raise NotImplementedError  # todo proper init like for linear
#                 h = int(in_features / 2)
#                 h2 = int(in_features / 4)
#                 # print(h,h2,out_features)
#                 layer1 = torch.nn.Linear(in_features, h)
#                 layer2 = torch.nn.Linear(h, h2)
#                 layer3 = torch.nn.Linear(h2, out_features)
#                 self._classification_layer = nn.Sequential(
#                     layer1,
#                     nn.Sigmoid(),
#                     layer2,
#                     nn.Sigmoid(),
#                     layer3)
#                 initializer(self._classification_layer)
#         else:
#             self.text_model = ConditionalLanguageModel(
#                 vocab=vocab,
#                 initializer=initializer,
#                 program_emb_size=embedding_dim,
#                 unconditional_lm=(self.task_setup_type == 'unconditional_lm'),
#                 eval_sanity_check_mode=text_eval_sanity_check_mode,
#                 hidden_dim=text_hidden_dim,
#                 embedding_dim=text_embedding_dim
#             )
#             # lstm/transformer generator
#             # note: 1) use bleu or other rewards also. otherwise stock names and stopwords will dominate
#             # -> compute using multiple annotations for that point;  create an eval class to make this easier
#             # see in the current setup which makes most sense
#             # --> stocknames and stopwords to be discouraged. verbs, adjectives, adverbs to be encouraged
#             # 2) using scheduled sampling
#             # 3) also look into self critical training -- being used in some current captioning setups
#
#         ### classification evals
#         self._accuracy = ConfusionMatrix(num_labels=num_labels)  # CategoricalAccuracy()
#         self._f1 = F1MeasureCustom()
#         if num_labels == 2:
#             self.nll_weights = torch.tensor([negative_class_wt, 1.0])
#         else:
#             self.nll_weights = None
#         self.num_labels = num_labels
#
#         ### text gen. evals
#         # bleu, perplexity, meteor, rouge, cider
#         self.use_distractors_evals = use_distractors_evals
#         self.distractor_evals = DistractorEvals()
#
#         ### others
#         self._prior_loss_tracker = Average()
#         self._clf_loss_tracker = Average()
#         self._reward_tracker = Average()
#         self._prior_loss_wt = prior_loss_wt
#         self._z_chosen_counts = [0 for j in range(self.num_programs)]
#         self._z_class_probs = [[Average() for k in range(self.num_labels)] for j in range(self.num_programs)]
#         self._z_class_wprob = [[Average() for k in range(self.num_labels)] for j in
#                                range(self.num_programs)]
#
#         ##
#         self._reinforce_moving_avg_baseline = None
#         if model_dict_load_file_path is not None:
#             print("==========>>>>>> Loading model from =", str(model_dict_load_file_path))
#             model_state = torch.load(model_dict_load_file_path)
#             print("model_state ===> ", model_state.keys())
#             # model_state_generator = {k[sz:]: v for k, v in model_state.items() if k[:sz] == '_generator.'}
#             # print("model_state = ", model_state)
#             self.load_state_dict(model_state)
#
#         ## Inference Network
#         self.inference_nw = InferenceModel( vocab=vocab,
#                                         hidden_dim=text_hidden_dim,
#                                         embedding_dim=text_embedding_dim,
#                                         initializer=initializer,
#                                         num_programs=num_labels,
#                                         arch_type='lstm')
#         print("self.inference_nw = ", self.inference_nw)
#
#
#     def get_proby_givenz(self, z_embedding, label=None, label_text=None, metadata=None, mode='ppl'):
#
#         if self.task_setup_type == 'classification':
#
#             #### this returns a distribution over labels
#             inp_to_predictor = z_embedding  # torch.cat()
#             # logit = (1.0/self.num_labels) * torch.ones(bs,self.num_labels) # #sanity check
#             logit = self._classification_layer(inp_to_predictor)
#             logprob_y_given_z = F.log_softmax(logit, dim=1)  # bs, num_labels
#             return logprob_y_given_z, {}
#
#         elif self.task_setup_type in ['text', 'gt_program_text']:
#
#             # print("[get_proby_givenz] label_text = ", label_text)
#             vals = self.text_model.forward(z_embedding, target_tokens=label_text, metadata=metadata, mode=mode)
#             logprob_ylabel_given_z = vals['logprob_ylabel_given_z']
#             return logprob_ylabel_given_z, vals
#             #### this return p(y=label|z)
#             # as a sanity check, can provide ground truth z, and see what the decoder learns
#
#         elif self.task_setup_type == 'unconditional_lm':
#
#             # print("uncondtional_lm: [get_proby_givenz] label_text = ", label_text)
#             vals = self.text_model.forward_unconditional_lm(target_tokens=label_text, metadata=metadata, mode=mode)
#             assert self.text_model.unconditional_lm
#             assert 'loss' in vals
#             logprob_y = vals['logprob_y']
#             return logprob_y, vals
#
#         else:
#             raise NotImplementedError
#
#     def marginalize_new(self, series, label, label_text,
#                         predictor_network, metadata,
#                         use_prior_potential: bool = False,
#                         train_unary_prior: bool = True):
#
#         bs = series.size()[0]
#         if train_unary_prior:
#             assert use_prior_potential
#
#         vals = predictor_network(series)
#         logits = vals['logits']  # bs, program_space
#         dist: torch.distributions.Categorical = torch.distributions.Categorical(logits=logits)
#
#         all_logprobs_list = []
#         all_scores_list = []
#
#         if self.use_inference_network:
#             posterior_over_program = self.inference_nw(label_text)
#
#         for z in range(self.num_programs):
#
#             action = torch.tensor(z)
#
#             program = self.programs[z]
#             # print("series = ", series)
#             _, score_z_x = program.forward(series, get_score_also=True)
#             all_scores_list.append(score_z_x.unsqueeze(0))
#             # print("[marginalize] ==> z = ", z, " ||  score_z_x = ", score_z_x,
#             #       " ||| label = ", label,
#             #       " ||| label_text=",label_text)
#
#             z_embedding = predictor_network.get_program_emb(action=action)  # embsize
#             z_embedding = z_embedding.unsqueeze(0).repeat(bs, 1)
#             logprob_y_given_z, _ = self.get_proby_givenz(z_embedding=z_embedding,
#                                                          label=label,
#                                                          label_text=label_text,
#                                                          metadata=metadata)
#             assert logprob_y_given_z.size()[0] == bs
#             logprob_y_given_z_numpy = logprob_y_given_z.data.cpu().numpy()
#             if self.task_setup_type != 'text':
#                 for logprob_y_given_z_numpy_zidx in logprob_y_given_z_numpy:
#                     for j, val in enumerate(logprob_y_given_z_numpy_zidx):
#                         self._z_class_probs[z][j](val)
#
#             # cur_prob = logprob_z.unsqueeze(1) + logprob_w_given_z.unsqueeze(1) + logprob_y_given_z # bs, num_labels
#             unnormalized_logprobz_givenx = score_z_x  # bs
#             if self.use_inference_network:
#                 pass
#             else:
#                 if self.task_setup_type != 'text':
#                     cur_prob_score = unnormalized_logprobz_givenx.unsqueeze(1) + logprob_y_given_z  # bs, num_labels
#                 else:
#                     cur_prob_score = unnormalized_logprobz_givenx + logprob_y_given_z  # bs
#
#             if use_prior_potential:
#                 logprob_z = dist.log_prob(action)  # bs   #.unsqueeze(0).repeat(bs,1) # bs
#                 if not train_unary_prior:
#                     logprob_z = logprob_z.detach()
#                 if self.task_setup_type != 'text':
#                     cur_prob_score += logprob_z.unsqueeze(1)  # bs, num_labels
#                 else:
#                     cur_prob_score += logprob_z  # bs
#
#
#
#             # print("[marginalize] ==> z = ", z, " ||  cur_prob = ", cur_prob_score )
#             all_logprobs_list.append(cur_prob_score.unsqueeze(0))  # 1, bs, numlabels or 1, bs
#
#         # computing the normalizing factor
#         # all_scores_list: numprograms,bs
#         # normalizing : bs
#         all_scores_list_tensor = torch.cat(all_scores_list, dim=0)  # nhum_programs, bs
#         all_scores_list_normalizer = torch.logsumexp(all_scores_list_tensor, dim=0)  # bs
#         all_scores_list_tensor_normalized = all_scores_list_tensor - all_scores_list_normalizer.unsqueeze(0)
#         # nhum_programs, bs
#         all_scores_list_normalizer = all_scores_list_normalizer.unsqueeze(1)  # bs,1
#
#         for z in range(self.num_programs):
#             if self.task_setup_type != 'text':
#                 all_logprobs_list[z] = all_logprobs_list[z] - all_scores_list_normalizer  # bs,numn_labels
#                 # and bs,1 -> bs,num_labels
#             else:
#                 all_logprobs_list[z] = all_logprobs_list[z] - all_scores_list_normalizer.squeeze(1)  # bs and bs,1 -> bs
#
#         if self.task_setup_type not in ['text', 'gt_text']: # todo - need to fix this. call only with prior scores
#             for z,all_scores_list_tensor_normalized_z in enumerate(all_scores_list_tensor_normalized):
#                 # print("all_scores_list_tensor_normalized_z : ", all_scores_list_tensor_normalized_z.size()) #bs
#                 # print("label : ", label.size()) #bs
#                 for val,labeli in zip(all_scores_list_tensor_normalized_z.data.cpu().numpy(),label.cpu().data.numpy()):
#                     self._z_class_wprob[z][labeli](math.exp(float(val)))
#
#         # prob(y|x) = sum_z (p(z)*p(y|x,z)*p(w=1|x,z)) = \logsumexp_z[ log p(z) + log p(y\x,z) + ..  ]
#         # print("[marginalize] all_logprobs_list = ", all_logprobs_list)
#         all_logprobs = torch.cat(all_logprobs_list, dim=0)
#         # all_logprobs_list: num_programs, bs, num_labels or num_programs, bs
#         total_logprob = torch.logsumexp(all_logprobs, dim=0)  # all_logprobs: bs, num_labels or bs
#         if self.task_setup_type == 'text':
#             avg_loss_for_ppl = torch.mean(-total_logprob)
#             self.text_model.log_ppl(avg_loss_for_ppl)
#
#         output_dict = {'total_logprob': total_logprob,
#                        'all_logprobs': all_logprobs.data.cpu().numpy()}
#
#         return output_dict
#
#     # inference
#     # condition on label text (and optionally the series)
#     # get approx posterior on the program distr.
#     # for now, sum over
#     # KL: prior needs to be computed
#     # so stesps
#     # 1) get approx posterior
#     # 2) marginalize over the approx posterior to compute conditional
#     # 3) KL loss computation
#     # p(z|x) \propto exp( f(z) * score(z,x) ). currently f(z) is not used.
#     # q(z|x,y) ...
#     # when f(z) is not used, eaxct KL computation would involve computing prior
#     # -- add a function to compute prior. will use in marginalize and KL
#     # -- add a function to marginalize over a given distribution. will use in inference and marginalize_new
#     # inference network:
#     # -> may be lstm or mean embedding for now
#     # test it out by trying to predict [1] class  [2] pattern and location both
#
#     # TODOs
#     # 1. inference n/w code
#     # 2. inference n/w test
#     # 3. prior computation function
#     # 4. marginalize function
#     # 5. verify
#     # 6. KL=0 run
#     # 7. Add KL computation
#     # 8. Test with 1 word, 2 word, complete text with fixed modules
#     # 9. Learn locate modules
#     # 10. Learn attend modules
#
#     def sample(self, series,
#                label,
#                label_text,
#                predictor_network,
#                use_argmax: bool = False,
#                metadata=None,
#                num_samples=1):
#
#         vals = predictor_network(series)
#         logits = vals['logits']  # bs, program_space
#         bs = series.size()[0]
#
#         dist: torch.distributions.Categorical = torch.distributions.Categorical(logits=logits)
#         all_logprobs_list = []
#         logprob_z_list = []
#
#         # for z in range(self.num_programs):
#         #     print("[sample] [NewThreeLayerLayoutPredictionProgram]: ", z, self.programs[z].__str__)
#         #     print("[sample] **Prior Distribution z=", z, torch.exp(dist.log_prob(torch.tensor(z))))
#         # print()
#         # print("[sample] label = ", label)
#         # print()
#
#         const = torch.ones(bs)
#         if torch.cuda.is_available():
#             const = const.cuda()
#         const = num_samples * const  # bs. all values are equal to K (num_samples)
#
#         for k in range(num_samples):
#
#             ##### sampling / argmax
#             if use_argmax:
#                 action, z_embedding, logprob_z = predictor_network.get_argmax(logits)
#                 assert num_samples == 1
#                 # print("[sample] mode = argmax || action = ", action)
#             else:
#                 action, z_embedding, logprob_z = predictor_network.sample(logits)
#             logprob_z_list.append(logprob_z)
#
#             zlist = list(action.data.cpu().numpy())  # bs
#             # print("[sample] zlist = ", zlist)
#             # print("[sample] logprob_z = ", logprob_z)
#             # print("[sample] series : ", series.size())
#             for zz in zlist:
#                 self._z_chosen_counts[zz] += 1
#
#             ##### compute program
#             program_list = [self.programs[z] for z in zlist]
#             logprob_w_given_z_list = [torch.log(programj(series[j:j + 1])) for j, programj in enumerate(program_list)]
#             logprob_w_given_z = torch.stack(logprob_w_given_z_list)  # bs,1
#             # print("[sample] ==> z = ", zlist, " ||  logprob_w_given_z = ", logprob_w_given_z, " \\\ label = ", label)
#             if self.task_setup_type != 'text':
#                 for zz, val, labelb in zip(zlist, logprob_w_given_z.cpu().data.numpy(), label.cpu().data.numpy()):
#                     self._z_class_wprob[zz][labelb](val)
#
#             ##### get output prob. given program
#             logprob_y_given_z_k, _ = self.get_proby_givenz(z_embedding=z_embedding,
#                                                            label=label,
#                                                            label_text=label_text,
#                                                            metadata=metadata)
#
#             #### computing cur_prob_k
#             cur_prob_k = logprob_w_given_z + logprob_y_given_z_k
#             all_logprobs_list.append(cur_prob_k.unsqueeze(0))
#
#         # logprob_y_given_z_numpy = logprob_y_given_z.data.cpu().numpy()
#         # assert len(logprob_y_given_z_numpy.shape)==2, logprob_y_given_z_numpy
#         # for zz,logprob_y_given_z_numpy_zidx in zip(zlist,logprob_y_given_z_numpy):
#         #     for j,val in enumerate(logprob_y_given_z_numpy_zidx):
#         #         self._z_class_probs[zz][j](val)
#         # print("[sample] ==> zlist = ", zlist, " ||  logprob_y_given_z = ", logprob_y_given_z)
#
#         # cur_prob = logprob_z.unsqueeze(1) + logprob_w_given_z.unsqueeze(1) + logprob_y_given_z # bs, num_labels
#         # logprob_y_w1_given_z = logprob_w_given_z + logprob_y_given_z # bs, num_labels
#         # print("[sample] ==> z = ", zlist, " ||  logprob_y_w1_given_z = ", logprob_y_w1_given_z)
#         # print()
#
#         all_logprobs = torch.cat(all_logprobs_list, dim=0)  # all_logprobs_list: num_programs, bs, num_labels
#         total_logprob = torch.logsumexp(all_logprobs, dim=0)  # total_logprob: bs, num_labels
#         # print("[sample]: total_logprob  : ", total_logprob.size())
#         total_logprob = total_logprob - torch.log(const).unsqueeze(1)  # subtract logK to normalize
#         # print("[sample] total_logprob= ", total_logprob)
#
#         # all_logprobs_list: num_samples, bs, num_labels
#         # logprob_z_list: num_samples, bs
#         logprob_z_list = torch.cat(logprob_z_list, dim=0)
#
#         output_dict = {'total_logprob': total_logprob,
#                        'logprob_z': logprob_z_list,
#                        'all_logprobs_list': all_logprobs,
#                        'type': 'multiple_samples'
#                        }
#
#         return output_dict
#
#     def unconditional_lm(self,
#                          label_text,
#                          metadata=None):
#
#         bs = label_text['tokens'].size()[0]
#
#         logprob_y, _ = self.get_proby_givenz(z_embedding=None,
#                                              label=None,
#                                              label_text=label_text,
#                                              metadata=metadata)
#         total_logprob = logprob_y
#
#         output_dict = {'total_logprob': total_logprob,
#                        'type': 'unconditional_lm'
#                        }
#
#         return output_dict
#
#     def gt_program_text(self,
#                         label_text,
#                         predictor_network,
#                         metadata,
#                         mode='ppl'):
#
#         # bs = label_text['tokens'].size()[0]
#         labels = [metadata_i['label']['labels'] for metadata_i in metadata]
#         if self.num_programs == 6:
#             mapper = LABEL_TO_PROGRAM_MAPPER_6programs
#         elif self.num_programs == 12:
#             mapper = LABEL_TO_PROGRAM_MAPPER_12programs
#         else:
#             raise NotImplementedError
#         gt_programs = [mapper[label] for label in labels]
#         gt_programs = np.array(gt_programs, dtype=np.long)
#         gt_programs = torch.tensor(gt_programs)
#         # print("[gt_program_text] labels = ", labels)
#         # print("[gt_program_text] label_text = ", label_text)
#         # print("[gt_program_text] gt_programs = ", gt_programs)
#         # print("[gt_program_text] gt_programs = ", [self.programs[g].__str__ for g in gt_programs.data.cpu().numpy()])
#         z_embedding = predictor_network.get_program_emb(gt_programs)
#
#         logprob_y, vals = self.get_proby_givenz(z_embedding=z_embedding,
#                                                 label=None,
#                                                 label_text=label_text,
#                                                 metadata=metadata,
#                                                 mode=mode)
#         total_logprob = logprob_y
#
#         output_dict = {'total_logprob': total_logprob,
#                        'type': 'gt_program_text'
#                        }
#         if mode == 'generate':
#             output_dict = {'generated_' + k: val for k, val in output_dict.items()}
#
#         return output_dict
#
#     def sample_new(self, series,
#                    label,
#                    label_text,
#                    predictor_network,
#                    use_argmax: bool = False,
#                    metadata=None,
#                    num_samples=1):
#         pass
#         # TODO
#
#     def generate(self, series,
#                  label,
#                  label_text,
#                  predictor_network,
#                  use_argmax: bool = False,
#                  metadata=None,
#                  use_prior_potential: bool = True):
#
#         bs = series.size()[0]
#         assert self.task_setup_type == 'text'
#
#         vals = predictor_network(series)
#         logits = vals['logits']  # bs, program_space
#         dist: torch.distributions.Categorical = torch.distributions.Categorical(logits=logits)
#
#         all_logprobs_list = []
#         all_scores_list = []
#
#         # for z in range(self.num_programs):
#         #     print("[marginalize] [NewThreeLayerLayoutPredictionProgram]: ", z, self.programs[z].__str__)
#         #     print("[marginalize] **Prior Distribution z=", z, torch.exp(dist.log_prob(torch.tensor(z))))
#         # print()
#         # print("label = ", label)
#         # print()
#
#         for z in range(self.num_programs):
#
#             action = torch.tensor(z)
#
#             program = self.programs[z]
#             _, score_z_x = program.forward(series, get_score_also=True)
#             all_scores_list.append(score_z_x.unsqueeze(0))
#             # print("[generate] ==> z = ", z, " ||  score_z_x = ", score_z_x,
#             #       " ||| label = ", label,
#             #       " ||| label_text=", label_text)
#
#             # cur_prob = logprob_z.unsqueeze(1) + logprob_w_given_z.unsqueeze(1) + logprob_y_given_z # bs, num_labels
#             unnormalized_logprobz_givenz = score_z_x  # bs
#             cur_prob_score = unnormalized_logprobz_givenz  # bs
#             if use_prior_potential:
#                 logprob_z = dist.log_prob(action)  # bs
#                 # print("[generate] ==> z = ", z, "logprob_z = ", logprob_z, " ||| label = ", label)
#                 cur_prob_score += logprob_z  # bs
#
#             # print("[generate] ==> z = ", z, " ||  cur_prob = ", cur_prob_score)
#             all_logprobs_list.append(cur_prob_score.unsqueeze(0))  # 1, bs
#             # all_logprobs_list.append(cur_prob.data.cpu().numpy())
#             # print()
#
#         # all_scores_list: numprograms,bs
#         all_scores_list_tensor = torch.cat(all_scores_list, dim=0)  # nhum_programs, bs
#
#         # -- now pick the one highest value
#         # -- get corresponding z_embeddings
#         # -- call get_proby_givenz with corresponding values - under 'generate' mode
#         # --
#
#         max_scores, z_argmax = all_scores_list_tensor.max(dim=0)  # all_scores_list_tensor is programs, bs
#         z_embedding = predictor_network.get_program_emb(z_argmax)  # bs,emb_dim
#         logprob_y_given_z, generate_vals = self.get_proby_givenz(z_embedding=z_embedding,
#                                                                  label=label,
#                                                                  label_text=label_text,
#                                                                  metadata=metadata,
#                                                                  mode='generate')  # logprob_y_given_z: bs
#
#         all_scores_list_normalizer = torch.logsumexp(all_scores_list_tensor, dim=0)  # bs
#         all_scores_list_normalizer = all_scores_list_normalizer  # bs
#         logprob_y_given_z = logprob_y_given_z - all_scores_list_normalizer
#
#         output_dict = {'generate_total_logprob': logprob_y_given_z,
#                        'generate_z_argmax': z_argmax.data.cpu().numpy()}
#
#         return output_dict
#
#     def classification_evals(self, label, tmp, output_dict):
#
#         # print("[classification_evals]===== ")
#         total_logprob = tmp['total_logprob']
#         bs = tmp['bs']
#         probs = torch.exp(total_logprob)
#         # print("[classification_evals] probs: ", probs.size())
#         # print("[classification_evals] probs = ", probs)
#
#         ##### compute decoder/classifier loss
#         # print("[classification_evals] label = ", label)
#         # TODO : in our case, the prob values don't sum to 1 because we are mdleling p(y,w=1|z)
#         # Not clear how nllLoss behaves in such cases
#         loss = F.nll_loss(total_logprob, target=label, weight=self.nll_weights)
#         # print("[classification_evals] total_logprob = ", total_logprob)
#         # print("[classification_evals] loss = ", loss)
#         loss *= self.clf_loss_wt
#         # print("[classification_evals] loss *= self.clf_loss_wt = ", loss)
#         output_dict["loss"] = loss
#         self._clf_loss_tracker(output_dict["loss"].data.cpu().item())
#
#         #### accuracy evals
#         # print("[classification_evals] predictions [0] = ", total_logprob.max(-1)[0])
#         # print("[classification_evals] predictions [1] = ", total_logprob.max(-1)[1])
#         # print("[classification_evals] label = ", label)
#         self._accuracy(total_logprob, label)  # fine to pass total_logprob as predictions
#         self._f1(total_logprob, label, probs=probs)  # fine to pass total_logprob as predictions
#
#         if self.model_type == 'prior_reinforce' and self.training:
#
#             # (self.training or self.test_time_method_type == 'sample'):
#
#             if tmp.get('type', 'single') == 'single':
#                 zlist = tmp['zlist']
#                 for j in range(label.size()[0]):
#                     pass
#                     # print("[classification_evals] j = ", j, "total_logprob[j]=", total_logprob[j])
#                     # print("[classification_evals] j = ", j, "label[j]=", label[j])
#                     # print("[classification_evals] j = ", j, "zlist[j]=", zlist[j])
#                     # print("[classification_evals] j = ", j, "loss=",
#                     #       F.nll_loss(total_logprob[j:j + 1], target=label[j:j + 1]))
#
#             logprob_z = tmp['logprob_z']
#
#             ##### compute preditor loss
#             # total_logprob: bs, num_labels
#             # these are probs p(y| x,z) * p(w=1|x,z) --> so won't sum to 1 when summing over y
#             # reward = probs.gather(1, label.view(-1, 1))  # get p_theta(y_gt) values -- bs,1
#
#             num_samples = None
#             if tmp.get('type', 'single') == 'single':
#                 reward = total_logprob.gather(1, label.view(-1, 1))  # get p_theta(y_gt) values -- bs,1
#                 # higher reward for the program which correctly predicts y and w=1
#             else:
#                 all_logprobs_list = tmp['all_logprobs_list']  # num_sample, bs, num_labels
#                 num_samples = all_logprobs_list.size()[0]  # .data.cpu().item()
#                 label_expanded = label.view(-1, 1)  # bs,1
#                 label_expanded = label_expanded.repeat(num_samples, 1)  # bs*num_samples, 1 :: bs,bs,bs,...
#                 # print("[classification_vals] - multiple samples - label = ", label)
#                 # print("[classification_vals] - multiple samples - label_expanded = ", label_expanded)
#                 # label_expanded = label_expanded.view(num_samples,bs,-1)
#                 # all_logprobs_list: num_samples, bs, num_labels
#                 # print("[classification_vals] - multiple samples - all_logprobs_list = ", all_logprobs_list)
#                 all_logprobs_list = all_logprobs_list.view(num_samples * bs,
#                                                            -1)  # :: bs,bs,bs... each of num_labels dimension
#                 # print("[classification_vals] - multiple samples - all_logprobs_list = ", all_logprobs_list)
#                 reward = all_logprobs_list.gather(1, label_expanded)  # num_labels*bs
#
#             reward = reward.detach()  # -- only prior should be updated through this term
#             # print("[classification_evals] **reward = ", reward)
#             # print("[classification_evals] **reward = ", reward.size())
#
#             if self.reinforce_baseline is not None:
#                 self._reward_tracker(torch.mean(reward).data.item())
#                 if self.reinforce_baseline == 'mean_std':
#                     reward = (reward - torch.mean(reward)) / (torch.std(reward) + 0.00000001)
#                     reward = reward.detach()
#                     # print("[classification_evals] reward after baseline adjustment = ", reward.data.cpu().numpy())
#                 elif self.reinforce_baseline == 'min_adjust':
#                     reward = (reward - torch.min(reward)) / (torch.max(reward) - torch.min(reward) + 0.00000001)
#                     reward = reward.detach()
#                     # print("[classification_evals] reward after baseline adjustment = ", reward.data.cpu().numpy())
#                 elif self.reinforce_baseline == 'moving_avg':
#                     if self._reinforce_moving_avg_baseline is None:
#                         self._reinforce_moving_avg_baseline = torch.mean(reward).detach()
#                     reward = reward - self._reinforce_moving_avg_baseline
#                     reward = reward.detach()
#                     self._reinforce_moving_avg_baseline = 0.99 * self._reinforce_moving_avg_baseline \
#                                                           + 0.01 * torch.mean(reward).detach()
#                     # print("[classification_evals] reward after baseline adjustment = ", reward.data.cpu().numpy())
#                     # print("[classification_evals] self._reinforce_moving_avg_baseline = ",
#                     #       self._reinforce_moving_avg_baseline.data.cpu().item())
#                 else:
#                     raise NotImplementedError
#
#             if tmp.get('type', 'single') == 'single':
#                 logprob_z = logprob_z.view(bs, 1)
#             else:  # multi sample case
#                 logprob_z = logprob_z.view(num_samples * bs, 1)
#             # print("[classification_evals] prior_objective_to_maximize:  logprob_z: ", logprob_z.size())
#             # print("[classification_evals] prior_objective_to_maximize:  logprob_z = ", logprob_z)
#             # print("[classification_evals] prior_objective_to_maximize:  rewardv: ", reward.size())
#             assert logprob_z.size() == reward.size()
#             # print("[classification_evals] prior_objective_to_maximize:  rewardv = ",
#             #       reward.data.cpu().numpy().reshape(-1))
#             # print("[classification_evals] prior_objective_to_maximize:  reward * logprob_z = ",
#             #       (reward * logprob_z).data.cpu().numpy().reshape(-1))
#             # print("[classification_evals] prior_objective_to_maximize:  reward * logprob_z = ", reward * logprob_z)
#             prior_objective_to_maximize = torch.mean(reward * logprob_z)
#             # print("[classification_evals] prior_objective_to_maximize = ", prior_objective_to_maximize)
#             prior_loss = -prior_objective_to_maximize
#
#             output_dict["prior_loss"] = self._prior_loss_wt * prior_loss
#             output_dict['loss'] += output_dict["prior_loss"]
#             self._prior_loss_tracker(output_dict["prior_loss"].data.cpu().item())
#
#         return output_dict
#
#     def text_evals(self, label_text, tmp, output_dict):
#         total_logprob = tmp['total_logprob']
#         # print(" *********************** total_logprob = ", total_logprob)
#         loss = - torch.mean(total_logprob)
#         output_dict["loss"] = loss
#         return output_dict
#
#     def forward(self, series: torch.FloatTensor,
#                 label: torch.LongTensor = None,
#                 label_text: [str, torch.LongTensor] = None,
#                 feats=None,
#                 distractors=None,
#                 metadata: List[Dict[str, Any]] = None):
#
#         predictor_network: EnumerateAllPrior = self.prior_model
#         bs = series.size()[0]
#
#         if self.task_setup_type == 'classification':
#             pass
#         else:
#             pass
#             # targets = label_text["tokens"]
#             # target_mask = get_text_field_mask(label_text)
#
#         tmp = {}
#
#         if self.training:
#
#             if self.task_setup_type == 'unconditional_lm':
#                 tmp.update(self.unconditional_lm(label_text=label_text,
#                                                  metadata=metadata))
#
#             elif self.task_setup_type == 'gt_program_text':
#                 tmp.update(self.gt_program_text(label_text=label_text,
#                                                 metadata=metadata, predictor_network=predictor_network))
#
#             else:
#
#                 if self.model_type == 'marginalize':
#                     tmp.update(self.marginalize(series, label, label_text, predictor_network, metadata=metadata))
#
#                 elif self.model_type == 'prior_reinforce':
#                     # compute predictor loss
#                     tmp.update(self.sample(series, label, label_text, predictor_network, metadata=metadata,
#                                            num_samples=self.reinforce_num_samples))
#                     # get reward
#                     # compute prior loss
#                     # prior loss. add to total
#
#                 elif self.model_type == 'inference_nw':
#                     raise NotImplementedError
#                     # get reward. this updates the posterior. add to total loss
#                     # get kl b.w prior and posteriror. add to total loss. this updates both prior and inference
#
#                 elif self.model_type == 'marginalize_new':
#                     tmp.update(self.marginalize_new(series, label, label_text,
#                                                     predictor_network,
#                                                     metadata=metadata,
#                                                     train_unary_prior=self.train_unary_prior,
#                                                     use_prior_potential=self.use_prior_potential))
#
#
#         else:
#
#             # test_time_method_type
#
#             if self.task_setup_type == 'unconditional_lm':
#                 tmp.update(self.unconditional_lm(label_text=label_text,
#                                                  metadata=metadata))
#
#             elif self.task_setup_type == 'gt_program_text':
#                 tmp.update(self.gt_program_text(label_text=label_text,
#                                                 metadata=metadata, predictor_network=predictor_network))
#                 generate_vals = self.gt_program_text(label_text=label_text,
#                                                      metadata=metadata, predictor_network=predictor_network,
#                                                      mode='generate')
#                 for k in generate_vals:
#                     assert k not in tmp
#                 tmp.update(generate_vals)
#
#             else:
#
#                 if self.test_time_method_type == 'marginalize':
#                     tmp.update(self.marginalize(series, label, label_text, predictor_network, metadata=metadata))
#
#                 elif self.test_time_method_type == 'sample':
#                     assert False  # not updated for latest changes in prior model
#                     # tmp.update(self.sample(series, label, label_text, predictor_network, metadata=metadata))
#
#                 elif self.test_time_method_type == 'marginalize_new':
#                     tmp.update(self.marginalize_new(series, label,
#                                                     label_text,
#                                                     predictor_network,
#                                                     metadata=metadata,
#                                                     train_unary_prior=self.train_unary_prior,
#                                                     use_prior_potential=self.use_prior_potential))
#
#                 elif self.test_time_method_type == 'argmax':
#                     assert False  # not updated for latest changes in prior model
#                     # tmp.update(self.sample(series, label, label_text,
#                     #                        predictor_network,
#                     #                        use_argmax=True,
#                     #                        metadata=metadata,
#                     #                        num_samples=1))
#
#                 if self.task_setup_type == 'text':
#
#                     generate_vals = self.generate(series, label, label_text, predictor_network, metadata=metadata)
#                     for k in generate_vals:
#                         assert k not in tmp
#                     tmp.update(generate_vals)
#
#                     if self.use_distractors_evals:
#                         assert self.test_time_method_type == 'marginalize_new'
#                         self.get_distractor_evals(series, distractors, predictor_network, metadata)
#
#         # print(">>>>>> metadata = >>>>>> ", metadata)
#
#         output_dict = {}
#
#         if self.task_setup_type == 'classification' and label is not None:
#             total_logprob = tmp['total_logprob']
#             tmp.update({'bs': bs})
#             output_dict = self.classification_evals(label=label, tmp=tmp, output_dict=output_dict)
#         elif self.task_setup_type in ['text', 'unconditional_lm', 'gt_program_text'] and label_text is not None:
#             output_dict = self.text_evals(label_text=label_text, tmp=tmp, output_dict=output_dict)
#             # total_logprob = tmp['total_logprob']
#             # output_dict = self.classification_evals(total_logprob, output_dict)
#
#         return output_dict
#
#     def get_distractor_evals(self, series: torch.FloatTensor,
#                              distractors,
#                              predictor_network,
#                              metadata):
#         bs = series.size()[0]
#         distractors = distractors['tokens']
#         for i, series_i in enumerate(series):
#             distractors_i = distractors[i]
#             # print("[get_distractor_evals : distractors_i = ", distractors_i)
#             # series_i: length
#             series_i = series_i.unsqueeze(0).repeat(len(distractors_i), 1)  # numdist, length
#             label_texti = {'tokens': distractors_i}
#             meta_i = [metadata[i] for _ in range(len(distractors_i))]
#             res_i = self.marginalize_new(series=series_i,
#                                          label=None,
#                                          label_text=label_texti,
#                                          predictor_network=predictor_network,
#                                          metadata=meta_i,
#                                          train_unary_prior=self.train_unary_prior,
#                                          use_prior_potential=self.use_prior_potential)
#             # ** Warning meta_i corresponds to correct one
#             # train_unary_prior = self.train_unary_prior,
#             # use_prior_potential = self.use_prior_potential)
#             total_logprob_vals_i = res_i['total_logprob'].data.cpu().numpy()
#             self.distractor_evals(total_logprob_vals_i)
#             # last one is the correct one. others are distractors
#
#             # gt_score = total_logprob_vals_i[-1]
#             # rank_gt = sorted(total_logprob_vals_i).find(gt_score)
#
#     def get_metrics(self, reset: bool = False) -> Dict[str, float]:
#
#         metrics = {}
#
#         if self.task_setup_type == 'classification':
#
#             if reset:
#                 print("CM = ", self._accuracy)
#
#             metrics.update(self._accuracy.get_metric(reset))  # {'accuracy': self._accuracy.get_metric(reset)}
#             metrics.update(self._f1.get_metric(reset))
#
#             for z in range(self.num_programs):
#                 for cld in range(self.num_labels):
#                     metrics.update({'class_' + str(cld) + '_z' + str(z) + '_wprob':
#                                         float(json.dumps(self._z_class_wprob[z][cld].get_metric(reset))) })
#                     # tmp = {'z_class_probs_' + str(z) + '_label_'+str(cld):
#                     #                     float(self._z_class_probs[z][cld].get_metric(reset))}
#                     # metrics.update(tmp)
#             # print(" ===>>> metrics = ", metrics)
#
#             if self.model_type in ['prior_reinforce', 'inference_nw']:
#                 prior_loss = self._prior_loss_tracker.get_metric(reset)
#                 reward = self._reward_tracker.get_metric(reset)
#                 clf_loss = self._clf_loss_tracker.get_metric(reset)
#                 metrics.update({'loss_prior': prior_loss, 'reward': reward, 'clf_loss': clf_loss})
#                 for z, cnt in enumerate(self._z_chosen_counts):
#                     # print("{'z_sample_cnt_'+str(z):cnt} = ", {'z_sample_cnt_' + str(z): cnt})
#                     metrics.update({'z_sample_cnt_' + str(z): cnt})
#
#             # print("metrics = ", metrics)
#             # for k, val in metrics.items():
#             #     print(k, type(val))
#             #     print(k, json.dumps(val))
#
#             if reset:
#                 self._z_chosen_counts = [0 for j in range(self.num_programs)]
#                 self._z_class_probs = [[Average() for k in range(self.num_labels)] for j in
#                                        range(self.num_programs)]
#                 self._z_class_wprob = [[Average() for k in range(self.num_labels)] for j in
#                                        range(self.num_programs)]
#
#
#         elif self.task_setup_type in ['text', 'unconditional_lm', 'gt_program_text']:
#
#             metrics.update(self.text_model.get_metrics(reset))
#             if self.use_distractors_evals:
#                 metrics.update(self.distractor_evals.get_metric(reset))
#
#         # if self.program_trainable_locate:
#         for locate_inst in self.instances['locate']:
#             metrics.update(locate_inst.get_useful_partitions())
#         # if self.program_trainable_attend:
#         for inst in self.instances['attend']:
#             metrics.update(inst.get_useful_partitions())
#
#         return metrics
#
#
# if __name__ == "__main__":
#     model = NewThreeLayerLayoutPredictionProgram(Vocabulary())
#     '''arr = torch.tensor([[0.0100, 0.0333, -0.0266, -0.0166, 0.0333, 0.0133, -0.0300,
#                          0.0033, 0.0100, 0.0333, -0.0266, -0.0166],
#                         [0.3100, 0.0333, -0.0266, -0.0166, 0.0333, 0.0133, -0.0300,
#                          0.3033, 0.0100, 0.0333, -0.0266, -0.0166]])
#     '''
#     arr = torch.tensor([[6, 9, 12, 15., 18, 18, 18, 19, 17, 19, 18, 18],
#                         [34, 35, 36, 35, 35, 42., 49, 56, 55, 57, 56, 56],
#                         [23, 24, 24, 23, 22, 23, 23, 23, 26, 29, 29, 30]
#                         ])
#     print("arr.size() = ", arr.size())
#     label = torch.tensor([1, 2, 3])
#     feats = model.forward(arr, label=label)
#     print(feats)
#     print()
#     print(model._accuracy.get_metric())
#
#     #
#     #
#     # def sample_old(self, series, label, label_text, predictor_network, use_argmax: bool=False, metadata=None):
#     #
#     #     vals = predictor_network(series)
#     #     logits = vals['logits']  # bs, program_space
#     #     bs = series.size()[0]
#     #
#     #     dist: torch.distributions.Categorical = torch.distributions.Categorical(logits=logits)
#     #     all_logprobs_list = []
#     #
#     #     for z in range(self.num_programs):
#     #         print("[sample] [NewThreeLayerLayoutPredictionProgram]: ", z, self.programs[z].__str__)
#     #         print("[sample] **Prior Distribution z=", z, torch.exp(dist.log_prob(torch.tensor(z))) )
#     #     print()
#     #     print("[sample] label = ", label)
#     #     print()
#     #
#     #     ##### sampling / argmax
#     #     if use_argmax:
#     #         action, z_embedding, logprob_z = predictor_network.get_argmax(logits)
#     #         print("[sample] mode = argmax")
#     #     else:
#     #         action, z_embedding, logprob_z = predictor_network.sample(logits)
#     #     zlist = list(action.data.cpu().numpy()) # bs
#     #     print("[sample] zlist = ", zlist)
#     #     print("[sample] logprob_z = ", logprob_z)
#     #     print("[sample] series : ", series.size())
#     #     for zz in zlist:
#     #         self._z_chosen_counts[zz]+=1
#     #
#     #     ##### compute program
#     #     program_list = [self.programs[z] for z in zlist]
#     #     logprob_w_given_z_list = [ torch.log(programj(series[j:j+1])) for j,programj in enumerate(program_list)]
#     #     logprob_w_given_z = torch.stack(logprob_w_given_z_list) # bs,1
#     #     print("[sample] ==> z = ", zlist, " ||  logprob_w_given_z = ", logprob_w_given_z, " \\\ label = ", label)
#     #     for zz,val,labelb in zip(zlist,logprob_w_given_z.cpu().data.numpy(),label.cpu().data.numpy()):
#     #         self._z_class_wprob[zz][labelb](val)
#     #
#     #     ##### get output prob. given program
#     #     logprob_y_given_z, _ = self.get_proby_givenz(z_embedding=z_embedding,
#     #                                               label=label,
#     #                                               label_text=label_text,
#     #                                               metadata=metadata)
#     #     # inp_to_predictor =  z_embedding # bs, embsize
#     #     # logit = self._classification_layer(inp_to_predictor) # bs, num_labels
#     #     # logprob_y_given_z = F.log_softmax(logit, dim=1) # bs, num_labels
#     #     logprob_y_given_z_numpy = logprob_y_given_z.data.cpu().numpy()
#     #     assert len(logprob_y_given_z_numpy.shape)==2, logprob_y_given_z_numpy
#     #     for zz,logprob_y_given_z_numpy_zidx in zip(zlist,logprob_y_given_z_numpy):
#     #         for j,val in enumerate(logprob_y_given_z_numpy_zidx):
#     #             self._z_class_probs[zz][j](val)
#     #     print("[sample] ==> zlist = ", zlist, " ||  logprob_y_given_z = ", logprob_y_given_z)
#     #
#     #     # cur_prob = logprob_z.unsqueeze(1) + logprob_w_given_z.unsqueeze(1) + logprob_y_given_z # bs, num_labels
#     #     logprob_y_w1_given_z = logprob_w_given_z + logprob_y_given_z # bs, num_labels
#     #     print("[sample] ==> z = ", zlist, " ||  logprob_y_w1_given_z = ", logprob_y_w1_given_z)
#     #     print()
#     #     total_logprob = logprob_y_w1_given_z #  bs, num_labels
#     #     # prob(y|x) = sum_z (p(z)*p(y|x,z)*p(w=1|x,z)) = \logsumexp_z[ log p(z) + log p(y\x,z) + ..  ]
#     #     print("[sample] total_logprob= ", total_logprob)
#     #
#     #     output_dict = {'total_logprob':total_logprob,
#     #                    'logprob_z':logprob_z,
#     #                    'zlist':zlist}
#     #
#     #     return output_dict
