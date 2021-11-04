from typing import Optional

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric

import numpy as np

# from allennlp.training.util import CONFIG_NAME
from typing import Any, Union, Dict, Iterable, List, Optional, Tuple
import json
from sklearn.metrics import precision_recall_fscore_support


@Metric.register("confusion_matrix")
class ConfusionMatrix(Metric):
    """
    Categorical Top-K accuracy. Assumes integer labels, with
    each item to be classified having a single correct class.
    Tie break enables equal distribution of scores among the
    classes with same maximum predicted scores.
    """

    def __init__(
        self, top_k: int = 1, tie_break: bool = False, num_labels: int = 2
    ) -> None:
        if top_k > 1 and tie_break:
            raise ConfigurationError(
                "Tie break in Categorical Accuracy "
                "can be done only for maximum (top_k = 1)"
            )
        if top_k <= 0:
            raise ConfigurationError("top_k passed to Categorical Accuracy must be > 0")
        self._top_k = top_k
        self._tie_break = tie_break
        self.correct_count = 0.0
        self.total_count = 0.0
        self._num_labels = num_labels
        self.cm = np.zeros((num_labels, num_labels))  # gt, prediction

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(
            predictions, gold_labels, mask
        )

        # Some sanity checks.
        num_classes = predictions.size(-1)
        if gold_labels.dim() != predictions.dim() - 1:
            raise ConfigurationError(
                "gold_labels must have dimension == predictions.size() - 1 but "
                "found tensor of shape: {}".format(predictions.size())
            )
        if (gold_labels >= num_classes).any():
            print("gold_labels = ", gold_labels)
            raise ConfigurationError(
                "A gold label passed to Categorical Accuracy contains an id >= {}, "
                "the number of classes.".format(num_classes)
            )

        predictions = predictions.view((-1, num_classes))
        gold_labels = gold_labels.view(-1).long()
        if not self._tie_break:
            # Top K indexes of the predictions (or fewer, if there aren't K of them).
            # Special case topk == 1, because it's common and .max() is much faster than .topk().
            if self._top_k == 1:
                top_k = predictions.max(-1)[1].unsqueeze(-1)
            else:
                top_k = predictions.topk(min(self._top_k, predictions.shape[-1]), -1)[1]

            # This is of shape (batch_size, ..., top_k).
            correct = top_k.eq(gold_labels.unsqueeze(-1)).float()
            # print(gold_labels.data.cpu().numpy().shape)
            # print(top_k.data.cpu().numpy().shape)
            for gl, pred in zip(
                gold_labels.data.cpu().numpy().reshape(-1),
                top_k.data.cpu().numpy().reshape(-1),
            ):
                # print(gl,pred)
                self.cm[gl][pred] += 1
        else:
            # prediction is correct if gold label falls on any of the max scores. distribute score by tie_counts
            max_predictions = predictions.max(-1)[0]
            max_predictions_mask = predictions.eq(max_predictions.unsqueeze(-1))
            # max_predictions_mask is (rows X num_classes) and gold_labels is (batch_size)
            # ith entry in gold_labels points to index (0-num_classes) for ith row in max_predictions
            # For each row check if index pointed by gold_label is was 1 or not (among max scored classes)
            correct = max_predictions_mask[
                torch.arange(gold_labels.numel()).long(), gold_labels
            ].float()
            tie_counts = max_predictions_mask.sum(-1)
            correct /= tie_counts.float()
            correct.unsqueeze_(-1)

        if mask is not None:
            correct *= mask.view(-1, 1).float()
            self.total_count += mask.sum()
        else:
            self.total_count += gold_labels.numel()
        self.correct_count += correct.sum()

    @overrides
    def get_metric(
        self, reset: bool = False
    ) -> Union[Dict[str, float], Dict[str, List[float]]]:
        """
        Returns
        -------
        The accumulated accuracy.
        """
        if self.total_count > 1e-12:
            accuracy = float(self.correct_count) / float(self.total_count)
        else:
            accuracy = 0.0
        # print(accuracy)
        ret = {"accuracy": accuracy}
        # for j in range(self._num_labels):
        #     ret['cm_' + str(j) ] = list(self.cm[j])
        #     # for k in range(self._num_labels):
        #     #     ret['cm_' + str(j) + str(k)] = self.cm[j][k]
        # print(ret)
        if reset:
            self.reset()
        return ret  # accuracy

    @overrides
    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0
        self.cm = np.zeros((self._num_labels, self._num_labels))  # gt, prediction

    def __str__(self):
        return json.dumps([list(v) for v in self.cm])
        # f"CocovalsMeasures(em={self._total_em}, f1={self._total_f1})"


import sklearn.metrics
from sklearn.metrics import roc_curve
from collections import Counter


@Metric.register("f1custom")
class F1MeasureCustom(Metric):
    def __init__(self, pos_label=1) -> None:
        self._predictions = []
        self._gt = []
        self._pos_label = pos_label
        self._probs = []

    @overrides
    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        probs: torch.Tensor = None,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(
            predictions, gold_labels, mask
        )

        # Some sanity checks.
        num_classes = predictions.size(-1)
        if gold_labels.dim() != predictions.dim() - 1:
            raise ConfigurationError(
                "gold_labels must have dimension == predictions.size() - 1 but "
                "found tensor of shape: {}".format(predictions.size())
            )
        if (gold_labels >= num_classes).any():
            raise ConfigurationError(
                "A gold label passed to Categorical Accuracy contains an id >= {}, "
                "the number of classes.".format(num_classes)
            )

        predictions = predictions.view((-1, num_classes))
        gold_labels = gold_labels.view(-1).long()
        top_k = predictions.max(-1)[1].unsqueeze(-1)
        # For each row check if index pointed by gold_label is was 1 or not (among max scored classes)
        correct = top_k.eq(gold_labels.unsqueeze(-1)).float()
        # print(" -----xxxx---- gold_labels = ", gold_labels)
        # print(" -----xxxx---- top_k = ", top_k)
        self._predictions.extend(top_k.data.cpu().numpy().reshape(-1))
        self._gt.extend(gold_labels.data.cpu().numpy())
        # print(" -----xxxx---- _predictions = ", self._predictions)
        # print(" -----xxxx---- _gt = ", self._gt)

        if probs is not None:
            probs_pos_class = probs[:, self._pos_label]
            self._probs.extend(list(probs_pos_class.data.cpu().numpy()))  #
        self._num_classes = num_classes

    @overrides
    def get_metric(self, reset: bool = False):  # -> Dict[str,Float]:
        num_classes = self._num_classes
        if num_classes > 2:
            # self.f1 = sklearn.metrics.f1_score(self._gt, self._predictions, average='micro')
            self.f1 = sklearn.metrics.f1_score(
                self._gt, self._predictions, average="macro"
            )
            self.prec, self.rec, _, _ = precision_recall_fscore_support(
                self._gt,
                self._predictions,
                pos_label=self._pos_label,
                labels=[self._pos_label],
            )
        else:
            self.f1 = sklearn.metrics.f1_score(
                self._gt, self._predictions, pos_label=self._pos_label
            )
            self.prec, self.rec, _, _ = precision_recall_fscore_support(
                self._gt,
                self._predictions,
                pos_label=self._pos_label,
                labels=[self._pos_label],
            )

        # print("self.prec, self.rec, = ", self.prec, self.rec, " f1 = ", self.f1)
        prediction_distribution = dict(Counter(self._predictions))
        gt_distribution = dict(Counter(self._gt))
        # print("gt_distribution = ", gt_distribution)
        auc_roc = -999
        f1_scores_max = -999
        threshold_max = -999

        if reset:  # to save computation time
            if num_classes == 2 and len(self._probs) > 0:
                fpr, tpr, thresholds = roc_curve(self._gt, self._probs)
                f1_scores = []
                for thresh in thresholds:
                    f1_scores.append(
                        sklearn.metrics.f1_score(
                            self._gt, [1 if m > thresh else 0 for m in self._probs]
                        )
                    )
                f1_scores = np.array(f1_scores)
                f1_scores_max = np.max(f1_scores)
                threshold_max = thresholds[np.argmax(f1_scores)]
                if len(gt_distribution) > 1:  # self._pos_label in gt_distribution:
                    auc_roc = sklearn.metrics.roc_auc_score(self._gt, self._probs)
                    # plt.figure()
                    # lw = 2
                    # plt.plot(fpr, tpr, color='darkorange',
                    #          lw=lw, label='ROC curve (area = %0.2f)' % auc_roc)
                    # plt.plot([ 0, 1 ], [ 0, 1 ], color='navy', lw=lw, linestyle='--')
                    # plt.xlim([ 0.0, 1.0 ])
                    # plt.ylim([ 0.0, 1.05 ])
                    # plt.xlabel('False Positive Rate')
                    # plt.ylabel('True Positive Rate')
                    # plt.title('Receiver operating characteristic example')
                    # plt.legend(loc="lower right")
                    # # plt.show()
                    # plt.savefig('roc_plot.png')
                else:
                    auc_roc = 0

        if reset:
            self.reset()
        ret = {
            "f1": self.f1,
            "rec": self.rec[0],
            "prec": self.prec[0],
            "auc_roc": float(auc_roc),
            "f1_scores_max": float(f1_scores_max),
            "threshold_max": float(threshold_max),
        }
        ret.update(
            {"f1_predlabel_" + str(k): v for k, v in prediction_distribution.items()}
        )
        ret.update({"f1_gtlabel_" + str(k): v for k, v in gt_distribution.items()})
        return ret

    @overrides
    def reset(self):
        self._predictions = []
        self._gt = []
        self._probs = []

    def __str__(self):
        return str(self.f1)


@Metric.register("f1customr")
class F1MeasureCustomEvalR:
    def __init__(self, pos_label=1, save_fig=True) -> None:
        self._predictions = []
        self._gt = []
        self._pos_label = pos_label
        self._probs = []
        self.save_fig = save_fig

    def __call__(self, label, score):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        print(label, score)
        self._gt.append(label)
        self._probs.append(score)

    # def get_metric(self, figname='roc_plot.png'):  # -> Dict[str,Float]:
    def get_metric(self, reset: bool = False):  # -> Dict[str,Float]:
        probs = np.array(self._probs)
        print(probs.shape)
        probs = (probs - probs.min()) / (probs.max() - probs.min())
        gt = np.array(self._gt)
        # print(gt.shape)
        # print(gt[:11])
        # print(probs.shape)
        # print(probs[:11])
        f1 = sklearn.metrics.f1_score(
            self._gt, self._predictions, pos_label=self._pos_label
        )

        auc_roc = 0
        if reset and len(probs) > 0:
            fpr, tpr, thresholds = roc_curve(gt, probs)
            f1_scores = []
            for thresh in thresholds:
                f1_scores.append(
                    sklearn.metrics.f1_score(
                        gt, [1 if m > thresh else 0 for m in probs]
                    )
                )
            f1_scores = np.array(f1_scores)
            f1_scores_max = np.max(f1_scores)
            threshold_max = thresholds[np.argmax(f1_scores)]
            auc_roc = sklearn.metrics.roc_auc_score(gt, probs)
            # print("auc_roc = ", auc_roc)
            # lw = 2
            # if self.save_fig:
            #     plt.plot(fpr, tpr, color='darkorange',
            #              lw=lw, label='ROC curve (area = %0.2f)' % auc_roc)
            #     plt.plot([ 0, 1 ], [ 0, 1 ], color='navy', lw=lw, linestyle='--')
            #     plt.xlim([ 0.0, 1.0 ])
            #     plt.ylim([ 0.0, 1.05 ])
            #     plt.xlabel('False Positive Rate')
            #     plt.ylabel('True Positive Rate')
            #     plt.title('Receiver operating characteristic example')
            #     plt.legend(loc="lower right")
            #     # plt.show()
            #     plt.savefig(figname)
        else:
            auc_roc = 0
            f1_scores_max = 0
        # print("self._gt Counter ===> ", Counter(gt))
        if reset:
            self.reset()
        return {"auc_roc": auc_roc, "f1_scores_max": f1_scores_max, "f1": f1}

    def reset(self):
        self._gt = []
        self._probs = []
