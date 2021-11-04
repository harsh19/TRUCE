from typing import Tuple

from overrides import overrides

# from allennlp.tools import squad_eval
from allennlp.training.metrics.metric import Metric

import os
import json
import numpy as np
import random


@Metric.register("distractor_evals")
class DistractorEvals(Metric):
    def __init__(self) -> None:
        self._ranks = []
        self._count = 0

    @overrides
    def __call__(self, scores, gt_pos=None):
        """
        Parameters
        ----------
        value : ``float``
            The value to average.
        """
        # gt_score = total_logprob_vals_i[gt_rank]
        if not gt_pos:
            gt_pos = len(scores) - 1
        print("[DistractorEvals]: scores = ", scores)
        scores = sorted(
            list(enumerate(scores)), key=lambda x: -x[1]
        )  # highers is better
        print("[DistractorEvals]: sorted scores = ", scores)
        rank = None
        j = 0
        for i, score in scores:
            j += 1
            if i == gt_pos:
                rank = j
                break
        self._ranks.append(rank)
        self._count += 1

    @overrides
    def get_metric(self, reset: bool = False):
        ret = {}
        print("[DistractorEvals]: self._ranks = ", self._ranks)
        if len(self._ranks) > 0:
            mrr = np.mean(np.array([1.0 / r for r in self._ranks]))
            acc = np.sum(np.array(self._ranks) == 1) * 1.0 / len(self._ranks)
            ret["distractor_evals_mrr"] = mrr
            ret["distractor_evals_acc"] = acc

        if reset:
            self.reset()

        return ret

    @overrides
    def reset(self):
        self._ranks = []
        self._count = 0

    def __str__(self):
        pass
