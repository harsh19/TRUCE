from typing import Tuple
from overrides import overrides
from allennlp.training.metrics.metric import Metric
import os
import json
import numpy as np
import random
import scipy
from collections import Counter


@Metric.register("diversity_evals")
class DiversityEvals(Metric):
    def __init__(self, model_name: str = None) -> None:
        self._unigrams = {"gt": [], "generated": []}
        self._bigrams = {"gt": [], "generated": []}
        self._trigrams = {"gt": [], "generated": []}
        self._generated_lst = []
        self._ctr = 1
        self._model_name = model_name

    @overrides
    def __call__(self, txt, typ):
        """
        Parameters
        ----------
        value : ``float``
            The value to average.
        """
        if typ == "gt" or typ == "generated":
            tokens = txt.strip().split()
            sz = len(tokens)
            for word in tokens:
                self._unigrams[typ].append(word)
            for j in range(sz - 1):
                self._bigrams[typ].append("_".join(tokens[j : j + 2]))
            for j in range(sz - 2):
                self._trigrams[typ].append("_".join(tokens[j : j + 3]))
            if typ == "generated":
                self._generated_lst.append(" ".join(tokens))
        else:
            assert False

    @overrides
    def get_metric(self, reset: bool = False):
        ret = {}
        for typ in ["generated", "gt"]:
            for ngram in ["unigrams", "bigrams", "trigrams"]:
                vals = self.__getattribute__("_" + ngram)
                if len(vals) == 0:
                    continue
                unigrams = dict(Counter(vals[typ]))
                unigrams = np.array(list(unigrams.values()))
                # print("unigrams = ", unigrams)
                unigrams = unigrams / np.sum(unigrams)
                entropy = scipy.stats.entropy(unigrams)
                uniq_cnt = len(unigrams)
                ret[typ + "_" + ngram + "_" + "entropy"] = entropy
                ret[typ + "_" + ngram + "_" + "uniq_cnt"] = uniq_cnt
            ret["generated_instances"] = len(self._generated_lst)
            ret["generated_uniq_cnt"] = len(set(self._generated_lst))
        if reset:
            if self._model_name is not None:
                fw = open(
                    "tmp/"
                    + self._model_name
                    + "/"
                    + str(self._ctr)
                    + ".generated_list.txt",
                    "w",
                )
                for s in list(set(self._generated_lst)):
                    fw.write(s + "\n")
                fw.close()
                self._ctr += 1
            self.reset()
        return ret

    @overrides
    def reset(self):
        self._unigrams = {"gt": [], "generated": []}
        self._bigrams = {"gt": [], "generated": []}
        self._trigrams = {"gt": [], "generated": []}
        self._generated_lst = []

    def __str__(self):
        pass
