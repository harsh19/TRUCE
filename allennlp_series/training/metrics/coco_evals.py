from typing import Tuple

from overrides import overrides

# from allennlp.tools import squad_eval
from allennlp.training.metrics.metric import Metric

from .pycocotools.coco import COCO
from .pycocoevalcap.eval import COCOEvalCap

import os
import json
import numpy as np

import logging
import sys

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
# logger.setLevel('INFO')

# import random
# import bert_score.bert_score as bert_score
# def test_multi_refs_working():
#     scorer = bert_score.BERTScorer(lang="en", batch_size=3, rescale_with_baseline=True)
#     cands = reader.bertscore_cands
#     refs = reader.bertscore_refs
#     P_mul, R_mul, F_mul = scorer.score(cands, refs,)
#     print("P_mul, R_mul, F_mul = ", P_mul, R_mul, F_mul)
#     print("P_mul, R_mul, F_mul = ", np.mean(P_mul.data.cpu().numpy()), np.mean(R_mul.data.cpu().numpy()), np.mean(F_mul.data.cpu().numpy()))


def bertscore_multi_refs_working(bert_scorer, bertscore_cands, bertscore_refs):
    scorer = bert_scorer  # bert_score.BERTScorer(lang="en", batch_size=3, rescale_with_baseline=True)
    # cands = ["I like lemons.", "Hi", "Hey", "Hello", "Go", ""]
    # refs = [
    #     ["I am proud of you.", "I like lemons.", "Go go go."],
    #     ["I am proud of you.", "Go go go."],
    #     ["Hi", ""],
    #     ["I am proud of you.", "I love lemons.", "Go go go.", "hello"],
    #     ["I am proud of you.", "Go go go.", "Go", "Go to school"],
    #     ["test"],
    # ]
    cands = bertscore_cands  # reader.bertscore_cands
    refs = bertscore_refs  # reader.bertscore_refs
    P_mul, R_mul, F_mul = scorer.score(
        cands,
        refs,
    )
    # print("P_mul, R_mul, F_mul = ", P_mul, R_mul, F_mul)
    # print("P_mul, R_mul, F_mul = ", np.mean(P_mul.data.cpu().numpy()), np.mean(R_mul.data.cpu().numpy()), np.mean(F_mul.data.cpu().numpy()))
    return {
        "prec": np.mean(P_mul.data.cpu().numpy()),
        "rec": np.mean(R_mul.data.cpu().numpy()),
        "f1": np.mean(F_mul.data.cpu().numpy()),
    }


@Metric.register("cocovals")
class CocovalsMeasures(Metric):
    def __init__(
        self, sanity_check_mode: bool = False, compute_bert_score: bool = False
    ) -> None:  # , cache_fname_prefix="tmp/") -> None:
        self._predictions = []
        self._gt = []
        self._count = 0
        self._predictions_imgs = {}
        self.sanity_check_mode = sanity_check_mode
        # self.cache_fname_prefix = cache_fname_prefix
        # self.ctr = 0
        self.compute_bert_score = compute_bert_score
        if compute_bert_score:
            import bert_score

            self.bert_scorer = bert_score.BERTScorer(
                lang="en", batch_size=3, rescale_with_baseline=True
            )
            # self.bert_scorer = bert_score.BERTScorer(lang="en", batch_size=3, rescale_with_baseline=False)

    @overrides
    def __call__(self, prediction, ground_truth, id):
        """
        Parameters
        ----------
        value : ``float``
            The value to average.
        """

        if id in self._predictions_imgs:
            logger.debug(f"Warning: Multiple predictions for the same id : id = {id} ")
            logger.debug(f"Warning: Ignoring current prediction for id = {id}")
        else:
            pred = {"image_id": id, "caption": prediction, "id": self._count}
            if self.sanity_check_mode:
                pred["caption"] = ground_truth
                print("Warning: OVERRIDING PREDICTION WITH GT FOR SANITY TESTING!!!")
            self._predictions.append(pred)
            self._predictions_imgs[id] = 1

        # tyipcally there will be multiple ground truth references
        gt = {"image_id": id, "caption": ground_truth, "id": self._count}
        self._gt.append(gt)

        self._count += 1

    @overrides
    def get_metric(self, reset: bool = False):

        ret = {}

        # print("[cocvals]: self._predictions: ", len(self._predictions))
        # print("[cocvals]: self._predictions[0]: ", self._predictions[0] )
        # print("[cocvals]: self._gt: ", len(self._gt))
        # print("[cocvals]: self._gt[0]: ", self._gt[0])
        # print("[cocvals]: images_list: ", len(list(set([ i['image_id'] for i in self._gt ]))) )

        if len(self._predictions) > 0:

            # pred_file = os.path.join(self.cache_fname_prefix + '_predfile.json')
            preds_lst = (
                self._predictions
            )  # [{'image_id': val[0], 'caption': val[1], 'id': val[0]} for i, val in
            # enumerate(self._predictions.items())]
            # json.dump(preds_lst, open(pred_file, 'w'))
            print("preds_lst = ", preds_lst)

            # gt_file = os.path.join(self.cache_fname_prefix + '_gtfile.json')
            gt_lst = self._gt
            print("gt_lst = ", gt_lst)
            # print(" gt_lst = ", gt_lst)
            images_list = list(set([i["image_id"] for i in gt_lst]))
            # [{'image_id': val[0], 'caption': val[1], 'id': val[0]} for i, val in enumerate(self._gt.items())]
            gt_lst_cocoformat = {
                "annotations": gt_lst,
                "images": [{"id": i} for i in images_list],
                "type": "captions",
                "info": "",
                "licenses": "",
            }
            # json.dump(gt_lst_cocoformat, open(gt_file, 'w'))

            coco = COCO(dataset=gt_lst_cocoformat)  # gt_file
            cocoRes = coco.loadRes(anns=preds_lst)  # pred_file)
            cocoEval = COCOEvalCap(coco, cocoRes)
            cocoEval.params["image_id"] = cocoRes.getImgIds()
            cocoEval.evaluate()

            for metric, score in cocoEval.eval.items():
                ret[metric] = score

            if self.compute_bert_score:
                img_to_pred = {i: "" for i in images_list}
                img_to_gt = {i: [] for i in images_list}
                for i in gt_lst:
                    img_to_gt[i["image_id"]].append(i["caption"])
                for i in preds_lst:
                    img_to_pred[i["image_id"]] = i["caption"]
                bertscore_refs = [img_to_gt[i] for i in images_list]
                # bertscore_cands = [img_to_gt[i][0] for i in images_list] # sanity check
                bertscore_cands = [img_to_pred[i] for i in images_list]
                info = bertscore_multi_refs_working(
                    self.bert_scorer,
                    bertscore_cands=bertscore_cands,
                    bertscore_refs=bertscore_refs,
                )
                print("info = ", info)
                ret.update({"bertscore_" + k: float(str(v)) for k, v in info.items()})

        if reset:
            self.reset()
        return ret

    @overrides
    def reset(self):
        self._predictions = []
        self._gt = []
        self._count = 0
        self._predictions_imgs = {}

    def __str__(self):
        pass  # return str(self.out) #f"CocovalsMeasures(em={self._total_em}, f1={self._total_f1})"
