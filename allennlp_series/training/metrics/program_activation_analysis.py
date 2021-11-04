from typing import Tuple

from overrides import overrides

# from allennlp.tools import squad_eval
from allennlp.training.metrics.metric import Metric

import os
import json
import numpy as np
import random
import scipy
from collections import Counter


@Metric.register("program_activation_evals")
class ProgramActivationEvals(Metric):
    def __init__(self, programs, perform_module_level_analysis: bool = True) -> None:
        self._predictions = {}
        self._predictions_gt = {}
        self._predictions_modules = {}
        self._predictions_gt_modules = {}
        self._programs = programs
        self._perform_module_level_analysis = perform_module_level_analysis

    @overrides
    def __call__(self, txt, typ, program_id):
        """
        Parameters
        ----------
        value : ``float``
            The value to average.
        """
        if typ == "generated":
            txt = " ".join(txt.strip().split())
            if program_id not in self._predictions:
                self._predictions[program_id] = []
            self._predictions[program_id].append(txt)
            if self._perform_module_level_analysis:
                loc = self._programs[program_id].locate.operator_name
                if loc not in self._predictions_modules:
                    self._predictions_modules[loc] = []
                self._predictions_modules[loc].append(txt)
                att = self._programs[program_id].attend.operator_name
                if att not in self._predictions_modules:
                    self._predictions_modules[att] = []
                self._predictions_modules[att].append(txt)
        elif typ == "gt":
            txt = " ".join(txt.strip().split())
            if program_id not in self._predictions_gt:
                self._predictions_gt[program_id] = []
            self._predictions_gt[program_id].append(txt)
            if self._perform_module_level_analysis:
                loc = self._programs[program_id].locate.operator_name
                if loc not in self._predictions_gt_modules:
                    self._predictions_gt_modules[loc] = []
                self._predictions_gt_modules[loc].append(txt)
                att = self._programs[program_id].attend.operator_name
                if att not in self._predictions_gt_modules:
                    self._predictions_gt_modules[att] = []
                self._predictions_gt_modules[att].append(txt)
        else:
            assert False

    @overrides
    def get_metric(self, reset: bool = False):
        ret = {}
        for program_id, predictions in self._predictions.items():
            all_tokens = []
            for s in predictions:
                all_tokens.extend(s.strip().split())
            ret["activationalaysis_" + str(program_id) + "_tokencount"] = len(
                all_tokens
            )
            ret["activationalaysis_" + str(program_id) + "_typescount"] = len(
                set(sorted(all_tokens))
            )
            print("[program_id] = ", program_id, "len(all_tokens) = ", len(all_tokens))
            print(
                "[program_id] = ",
                program_id,
                "len(set(sorted(all_tokens)) = ",
                len(set(sorted(all_tokens))),
            )
            print(
                "[program_id] = ",
                program_id,
                " SortedByCounts = ",
                sorted(list(dict(Counter(all_tokens)).items()), key=lambda x: x[1]),
            )
        for program_id, predictions in self._predictions_gt.items():
            all_tokens = []
            for s in predictions:
                all_tokens.extend(s.strip().split())
            ret["activationalaysis_GT_" + str(program_id) + "_tokencount"] = len(
                all_tokens
            )
            ret["activationalaysis_GT_" + str(program_id) + "_typescount"] = len(
                set(sorted(all_tokens))
            )
            print(
                "[program_id] = ", program_id, "GT: len(all_tokens) = ", len(all_tokens)
            )
            print(
                "[program_id] = ",
                program_id,
                "GT: len(set(sorted(all_tokens)) = ",
                len(set(sorted(all_tokens))),
            )
            print(
                "[program_id] = ",
                program_id,
                "GT: SortedByCounts = ",
                sorted(list(dict(Counter(all_tokens)).items()), key=lambda x: x[1]),
            )
        if self._perform_module_level_analysis:
            for module_id, predictions in self._predictions_modules.items():
                all_tokens = []
                for s in predictions:
                    all_tokens.extend(s.strip().split())
                ret["activationalaysis_" + str(module_id) + "_tokencount"] = len(
                    all_tokens
                )
                ret["activationalaysis_" + str(module_id) + "_typescount"] = len(
                    set(sorted(all_tokens))
                )
                print(
                    "[module_id] = ", module_id, "len(all_tokens) = ", len(all_tokens)
                )
                print(
                    "[module_id] = ",
                    module_id,
                    "len(set(sorted(all_tokens)) = ",
                    len(set(sorted(all_tokens))),
                )
                print(
                    "[module_id] = ",
                    module_id,
                    " SortedByCounts = ",
                    sorted(list(dict(Counter(all_tokens)).items()), key=lambda x: x[1]),
                )
            for module_id, predictions in self._predictions_gt_modules.items():
                all_tokens = []
                for s in predictions:
                    all_tokens.extend(s.strip().split())
                ret["activationalaysis_GT_" + str(module_id) + "_tokencount"] = len(
                    all_tokens
                )
                ret["activationalaysis_GT_" + str(module_id) + "_typescount"] = len(
                    set(sorted(all_tokens))
                )
                print(
                    "[module_id] = ",
                    module_id,
                    "GT: len(all_tokens) = ",
                    len(all_tokens),
                )
                print(
                    "[module_id] = ",
                    module_id,
                    "GT: len(set(sorted(all_tokens)) = ",
                    len(set(sorted(all_tokens))),
                )
                print(
                    "[module_id] = ",
                    module_id,
                    "GT: SortedByCounts = ",
                    sorted(list(dict(Counter(all_tokens)).items()), key=lambda x: x[1]),
                )
        if reset:
            self.reset()
        return ret

    @overrides
    def reset(self):
        self._predictions = {}
        self._predictions_gt = {}
        self._predictions_modules = {}
        self._predictions_gt_modules = {}

    def __str__(self):
        pass
