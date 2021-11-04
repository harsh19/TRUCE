import json
import logging
import numpy as np
from typing import Dict
from overrides import overrides
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import ArrayField, MetadataField, ListField, LabelField
from allennlp.data.fields.text_field import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, WordTokenizer  # , WhitespaceTokenizer

# from allennlp_series.data.analysis.entity_detection import EntityDetectionOverlap, EntityDetectionAll
# from allennlp_series.training.metrics import F1MeasureCustom, ExplanationEval, PrecisionEval
import string
from allennlp_series.common.constants import *
import copy
from allennlp.data.tokenizers.word_stemmer import PorterStemmer, PassThroughWordStemmer
from allennlp_series.training.metrics import CocovalsMeasures
import random
from collections import Counter
from nltk.corpus import stopwords
logger = logging.getLogger(__name__)


word_pair_labels_factorized_tokeep = []
heuristic_label_mapper = {
    wp: i for i, wp in enumerate(word_pair_labels)
}  # this is changed in constructor
heuristic_labels_list = []

stw = stopwords.words("english")

def remove_s(w):
    if w[-1] != "s":
        return w
    return w[:-1]

def get_pairs(s, mapper=None):
    # print(" || s = ", s)
    s = [remove_s(w) for w in s if w not in stw]
    ret = []
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            ret.append("_".join(sorted([s[i], s[j]])))
    # print("--- ret = ", ret)
    return ret

def normalize_text(s):
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    # s = START_SYMBOL + ' ' + s  + ' ' +  END_SYMBOL
    return s.strip()


@DatasetReader.register("stock_series_text_reader")
class StockDataTextReader(DatasetReader):
    def __init__(
        self,
        # tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_pieces: int = 512,
        debug: bool = False,
        train_data_fraction: float = 1.0,
        perform_stemming: bool = False,
        # label_data_subset_type: str = None,
        overfit_mode: bool = False,
        single_word_label_mode: bool = False,
        single_word_label_type: str = "one",
        self_eval_mode: bool = False,
        prepare_bert_score_data: bool = False,
        use_heuristic_labels: bool = False,
        heuristic_label_mapper_type: str = "normal",
        num_attend_modules: int = None,
        word_pair_labels_factorized_tokeep_type: str = "all",
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        self._perform_stemming = perform_stemming
        if perform_stemming:
            self._stemmer = PorterStemmer()
        else:
            self._stemmer = PassThroughWordStemmer()
        self._tokenizer = WordTokenizer(
            start_tokens=[START_SYMBOL],
            end_tokens=[END_SYMBOL],
            word_stemmer=self._stemmer,
        )
        # or WordSplitter() #or BertBasicWordSplitter()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._max_pieces = max_pieces
        self.debug_mode = debug  # True, #True, #debug
        self._train_data_fraction = train_data_fraction
        self.overfit_mode = overfit_mode
        self.file_record = {}
        self.single_word_label_mode = single_word_label_mode
        self.self_eval_mode = self_eval_mode
        if self.self_eval_mode:
            self.ngram_overlap_eval = CocovalsMeasures()
        self.single_word_label_type = single_word_label_type
        assert single_word_label_type in ["one", "two", "two_reversed"]
        self.prepare_bert_score_data = prepare_bert_score_data
        if self.prepare_bert_score_data:
            self.fw_bert_score = [
                open("bert_score/example/stock_data/hyps.txt", "w"),
                open("bert_score/example/stock_data/refs.txt", "w"),
                open("bert_score/example/stock_data/refs2.txt", "w"),
            ]
            self.bertscore_cands = []
            self.bertscore_refs = []
        self.use_heuristic_labels = use_heuristic_labels
        self.heuristic_label_mapper_type = heuristic_label_mapper_type
        self.word_pair_labels_factorized_tokeep_type = (
            word_pair_labels_factorized_tokeep_type
        )
        global heuristic_label_mapper
        global word_pair_labels
        global word_pair_labels_factorized_tokeep
        global word_pair_labels_factorized
        # word_pair_labels_touse = word_pair_labels
        if self.use_heuristic_labels:
            if heuristic_label_mapper_type == "normal":
                # word_pair_labels = word_pair_labels
                raise NotImplementedError
                # pass
            elif heuristic_label_mapper_type == "v2":
                raise NotImplementedError
                # word_pair_labels = word_pair_labels_v2
                # heuristic_label_mapper = {
                #     wp: i for i, wp in enumerate(word_pair_labels)
                # }
            elif heuristic_label_mapper_type in ["factorized", "factorizedv2"]:
                if heuristic_label_mapper_type == "factorizedv2":
                    word_pair_labels_factorized = word_pair_labels_factorizedv2
                if self.word_pair_labels_factorized_tokeep_type == "all":
                    for locate_id, v in enumerate(
                        word_pair_labels_factorized["locate"]
                    ):
                        for attend_id, a in enumerate(
                            word_pair_labels_factorized["pattern"]
                        ):
                            tmp = "_".join(sorted([v, a]))
                            word_pair_labels_factorized_tokeep.append(tmp)
                elif self.word_pair_labels_factorized_tokeep_type == "small1":
                    word_pair_labels_factorized_tokeep = [
                        "_".join(sorted(["increase", "begin"])),
                        "_".join(sorted(["decrease", "middle"])),
                        "_".join(sorted(["flat", "end"])),
                    ]
                elif self.word_pair_labels_factorized_tokeep_type == "small2":
                    word_pair_labels_factorized_tokeep = [
                        "_".join(sorted(["increase", "begin"])),
                        "_".join(sorted(["decrease", "begin"])),
                        "_".join(sorted(["flat", "begin"])),
                        "_".join(sorted(["decrease", "middle"])),
                        "_".join(sorted(["increase", "end"])),
                    ]
                elif self.word_pair_labels_factorized_tokeep_type == "small3":
                    word_pair_labels_factorized_tokeep = [
                        "_".join(sorted(["increase", "begin"])),
                        "_".join(sorted(["increase", "middle"])),
                        "_".join(sorted(["decrease", "begin"])),
                        "_".join(sorted(["decrease", "middle"])),
                        "_".join(sorted(["flat", "begin"])),
                        "_".join(sorted(["flat", "middle"])),
                    ]
                else:
                    raise NotImplementedError
                heuristic_label_mapper = {}
                word_pair_labels = []
                for locate_id, v in enumerate(word_pair_labels_factorized["locate"]):
                    locate_id_touse = locate_id + 1  # avoiding 0,0
                    for attend_id, a in enumerate(
                        word_pair_labels_factorized["pattern"]
                    ):
                        id = locate_id_touse * num_attend_modules + attend_id
                        # this emumeration should be consistent to the enumeration in the models.py
                        tmp = "_".join(sorted([v, a]))
                        if tmp not in word_pair_labels_factorized_tokeep:
                            continue
                        heuristic_label_mapper[tmp] = id
                        word_pair_labels.append(tmp)
                assert 0 not in heuristic_label_mapper.values()
                # heuristic_label_mapper['none'] = 0
                print("heuristic_label_mapper = ", heuristic_label_mapper)
            else:
                raise NotImplementedError

    @overrides
    def _read(self, file_path: str):  # processed_data/pilot12val.json

        ##############################################
        print()
        print("=" * 83)

        data_label_dist = {}

        if file_path.count("train") > 0:
            assert "train" not in self.file_record
            self.file_record["train"] = file_path
        elif file_path.count("dev") > 0:
            assert "dev" not in self.file_record
            self.file_record["dev"] = file_path

        if self.overfit_mode:
            if file_path.count("dev") > 0 or file_path.count("val") > 0:  # dev split
                file_path = self.file_record["train"]
                print("#" * 22)
                print(" FORCE CHANGING DEV FILE TO ", file_path)
                print("#" * 22)

        self._instance_num = 0
        skip_cnt = 0
        print("---->>>>>>>>>>>>>>>>>>>>>>> Reading instances from file at: ", file_path)
        data = json.load(open(file_path, "r"))

        sz = len(data)  # number of images
        image_ids = list(data.keys())
        print("sz : len(image_ids) = ", sz)

        if file_path.count("train") > 0 and self._train_data_fraction < 1.0:
            new_sz = int(sz * self._train_data_fraction)
            image_ids = image_ids[:new_sz]
        print(
            "train_data_fraction: ",
            self._train_data_fraction,
            " len(image_ids) = ",
            len(image_ids),
        )

        ##############################################

        inst_num = 0

        for i, img_id in enumerate(image_ids):

            if self.debug_mode:
                pass
                # print("*********>>>>>>> img_id = ", img_id)

            row = data[img_id]
            series = row["series"]
            idx = row["idx"]
            meta = row["meta"]
            meta.update({"idx": idx})  # data collection idx is the reference idx now on

            annotations = row["annotations"]
            for j, annotation in enumerate(annotations):
                meta = copy.deepcopy(meta)
                meta["raw_text"] = annotation
                txt = annotation  # [0]
                yield self.text_to_instance(series, txt, meta)
                inst_num += 1
                if self.debug_mode:
                    break

            if self.self_eval_mode:
                annotations_id = copy.deepcopy(row["annotations"])
                random.shuffle(annotations_id)
                predicted_str = annotations_id[0]
                for target_str in annotations_id[1:]:
                    print(
                        "predicted_str, target_str,  = ",
                        predicted_str,
                        target_str,
                    )
                    self.ngram_overlap_eval(predicted_str, target_str, img_id)
                if self.prepare_bert_score_data:
                    self.fw_bert_score[0].write(annotations_id[0].strip() + "\n")
                    self.fw_bert_score[1].write(annotations_id[1].strip() + "\n")
                    self.fw_bert_score[2].write(annotations_id[2].strip() + "\n")
                    self.bertscore_cands.append(annotations_id[0].strip())
                    self.bertscore_refs.append(
                        [annotations_id[1].strip(), annotations_id[2].strip()]
                    )

            if self.debug_mode and inst_num > 1:
                break

        print(
            "file_path = ",
            file_path,
            " || skip_cnt = ",
            skip_cnt,
            " || inst_num = ",
            inst_num,
        )
        print("file_path = ", file_path, " ||| data_label_dist : ", data_label_dist)

    @overrides
    def text_to_instance(self, series, label_text, meta=None) -> Instance:
        fields = {}
        series = np.array(series)
        fields["series"] = ArrayField(series)
        label_text = normalize_text(label_text)
        if self.use_heuristic_labels:
            txt = label_text  # [str(s) for s in txt]
            word_pairs = get_pairs(txt.strip().split())
            labels = [wp for wp in word_pairs if wp in word_pair_labels]
            if self.debug_mode:
                print("labels = ", labels)
                print("word_pair_labels = ", word_pair_labels)
            if len(labels) == 1:
                heuristic_label = labels[0]
                heuristic_label = heuristic_label_mapper[heuristic_label]
            else:
                heuristic_label = 0
            heuristic_labels_list.append(heuristic_label)
            fields["heuristic_label"] = LabelField(
                heuristic_label, label_namespace="heuristic_label", skip_indexing=True
            )
        # label_text = self._tokenizer.split_words(label_text)
        label_text = self._tokenizer.tokenize(label_text)
        fields["label_text"] = TextField(label_text, self._token_indexers)
        if self.debug_mode:
            print("meta = ", meta)
        metadata = meta
        fields["metadata"] = MetadataField(metadata)
        ins = Instance(fields=fields)
        if self.debug_mode:
            print("=====>>>> ins = ", ins)
        return ins




if __name__ == "__main__":

    # reader = SyntheticDataReader(tokenizer=None,
    #                              debug=True)
    # cnt = 0
    # for ins in reader._read("../data_collection/tables/synthetic_data/data1_10k_12r_dev.pkl"):
    #     print("ins = ", ins)
    #     print("cnt = ", cnt)
    use_heuristic_labels = True
    reader = StockDataTextReader(
        debug=False,
        # label_data_subset_type="all",  #''all_but_throughout',
        perform_stemming=False,  # True,
        # add_distractors_data=False,
        self_eval_mode=True,
        prepare_bert_score_data=True,
        use_heuristic_labels=use_heuristic_labels,
        heuristic_label_mapper_type="factorized",
        num_attend_modules=6,
    )
    cnt = 0
    # for ins in reader._read("processed_data/pilot16aval.json"):
    for ins in reader._read("processed_data/pilot16bval.json"):
    # for ins in reader._read("processed_data/pilot16cval.json"):
        # print("cnt = ", cnt)
        # print("ins = ", ins)
        # print("ins: meta = ", ins.fields["metadata"])
        cnt += 1
        # print("=" * 31)
    print("cnt = ", cnt)

    ctr = Counter(heuristic_labels_list)
    print("heuristic_labels_list : ", ctr)
    print("heuristic_labels distribution : ", ctr[0]/( sum(ctr.values())) )
    # heuristic_labels_list :  Counter({0: 460, 18: 26, 12: 25, 19: 23, 20: 14, 14: 14, 21: 11, 6: 9, 15: 9, 13: 8,
    # 27: 6, 7: 6, 8: 5, 24: 5, 9: 4, 25: 4, 26: 1})
    # heuristic_labels distribution :  0.6912280701754386 -> this much data is untagged

    # most freq words: ('ends', 263), ('near', 267), ('peaks', 271), ('throughout', 276), ('steadily', 280), ('sharp', 292), ('is', 311), ('from', 330), ('value', 333), ('start', 335), ('a', 353), ('steady', 389), ('decreases', 415), ('increase', 453), ('flat', 454), ('increases', 540), ('beginning', 679), ('middle', 1415), ('in', 1497), ('end', 1532), ('at', 1656), ('the', 3614)]
    # end,middle,begin,throughout(is not in the most freq word list; comes behind steadily, sharp, start, value; but prob is 4th most freq locate intent bearing word)
    # increase,decrease,peak,flat makes sense

    # heuristic_label_mapper =  {'beginning_increase': 6, 'beginning_decrease': 7, 'beginning_peak': 8, 'beginning_flat': 9, 'increase_middle': 12, 'decrease_middle': 13, 'middle_peak': 14, 'flat_middle': 15, 'end_increase': 18, 'decrease_end': 19, 'end_peak': 20, 'end_flat': 21, 'increase_throughout': 24, 'decrease_throughout': 25, 'peak_throughout': 26, 'flat_throughout': 27}

    metrics = reader.ngram_overlap_eval.get_metric(reset=True)
    print("metrics = ", metrics)

    # ****** Random Prediction
    # reader._self_eval_random("processed_data/pilot13val.json")
    # metrics = reader.ngram_overlap_eval.get_metric(reset=True)
    # print("metrics = ", metrics)

    # ****** stats
    # reader._self_eval_random("processed_data/pilot16aval.json")
    # reader._self_eval_random("processed_data/pilot16cval.json")
    vals = {}
    all_tokens = []
    import scipy.stats

    for split in ['train','val']:
        inst_cnt = 0
        ctr = {}
        lengths = []
        for inst in reader.read("processed_data/pilot16c"+split+".json"):
            print("inst['metadata'].metadata = ", inst['metadata'].metadata )
            #label = inst['metadata'].metadata['label']['labels']
            #print('label = ', label)
            #ctr[label] = ctr.get(label,0) + 1
            print("inst['label_text'].tokens = ", inst['label_text'].tokens)
            lengths.append(len(inst['label_text'].tokens)-2)
            all_tokens.extend(inst['label_text'].tokens[1:-1])
            inst_cnt += 1
        print("==========")
        print("inst_cnt = ", inst_cnt)
        vals[split] = {}
        vals[split]['inst_cnt'] = inst_cnt
        vals[split]['ctr'] = ctr
        vals[split]['lengths'] = Counter(lengths)
        vals[split]['lengths-summary'] = scipy.stats.describe(lengths)
    print("---------------")
    print("vals = ", vals)
    # print("vals = ", json.dumps(vals, indent=2))
    print("="*21)
    print("-----VOCABULARY------")
    all_tokens = [str(s) for s in all_tokens]
    print(type(all_tokens[0]))
    print("all_tokens[:5] = ", all_tokens[:5])
    print("#count = ", len(all_tokens))
    print("#Vocab-size = ", len(set(sorted(all_tokens))))
    print("#SortedByCounts = ", sorted( list(dict(Counter(all_tokens)).items()), key=lambda x:x[1]) )
    fw = open('stock_data_vocab.tsv','w')
    for kv in  sorted( list(dict(Counter(all_tokens)).items()), key=lambda x:x[1]):
        k,v = kv
        fw.write('\t'.join([str(k),str(v)]))
        fw.write('\n')
    fw.close()
    # vals = {'train': {'inst_cnt': 5040, 'ctr': {},
    #                   'lengths': Counter({5: 1792, 4: 1323, 6: 713, 3: 453, 7: 315, 8: 248, 2: 183, 9: 6, 0: 4, 1: 3}),
    #                   'lengths-summary': DescribeResult(nobs=5040, minmax=(0, 9), mean=4.861309523809524,
    #                                                     variance=1.8126722280497827, skewness=0.2884459453944344,
    #                                                     kurtosis=0.3195253305702921)},
    #         'val': {'inst_cnt': 630, 'ctr': {},
    #                 'lengths': Counter({5: 235, 4: 169, 6: 84, 3: 50, 7: 41, 8: 36, 2: 14, 9: 1}),
    #                 'lengths-summary': DescribeResult(nobs=630, minmax=(2, 9), mean=4.947619047619048,
    #                                                   variance=1.7190324778560075, skewness=0.47368943427977817,
    #                                                   kurtosis=0.3073705096169568)}}
    # == == == == == == == == == == =
    # -----VOCABULARY - -----
    # <
    #
    # class 'str'>
    #
    #
    # all_tokens[:5] = ['decreases', 'smoothly', 'in', 'the', 'beginning']
    # # count =  27618
    # # Vocab-size =  861


    # if reader.prepare_bert_score_data:
    #     for fwb in reader.fw_bert_score:
    #         fwb.close()
    #
    #     # import bert_score.bert_score as bert_score
    #     import bert_score #.bert_score as bert_score
    #     def test_multi_refs_working():
    #         scorer = bert_score.BERTScorer(lang="en", batch_size=3, rescale_with_baseline=True)
    #         # cands = ["I like lemons.", "Hi", "Hey", "Hello", "Go", ""]
    #         # refs = [
    #         #     ["I am proud of you.", "I like lemons.", "Go go go."],
    #         #     ["I am proud of you.", "Go go go."],
    #         #     ["Hi", ""],
    #         #     ["I am proud of you.", "I love lemons.", "Go go go.", "hello"],
    #         #     ["I am proud of you.", "Go go go.", "Go", "Go to school"],
    #         #     ["test"],
    #         # ]
    #         cands = reader.bertscore_cands
    #         refs = reader.bertscore_refs
    #         P_mul, R_mul, F_mul = scorer.score(cands, refs,)
    #         print("P_mul, R_mul, F_mul = ", P_mul, R_mul, F_mul)
    #         print("P_mul, R_mul, F_mul = ", np.mean(P_mul.data.cpu().numpy()), np.mean(R_mul.data.cpu().numpy()), np.mean(F_mul.data.cpu().numpy()))
    #     test_multi_refs_working()

    # ****** check weak labels
    all_texts = []
    import nltk
    from nltk.corpus import stopwords

    stw = stopwords.words("english")
    mapper = (
        {}
    )  # {'increases':'increase','decreases':'decrease','dips':'dip','peaks':'peak','declines':'decline'}

# all_word_pairs_items_top =  ['increase_middle', 'decrease_end', 'end_value', 'end_increase', 'middle_peak', 'end_start', 'end_toward', 'beginning_increase', 'flat_stay', 'middle_rise', 'end_near', 'end_peak', 'decrease_middle', 'beginning_decrease', 'end_lower', 'lowest_point', 'end_slightly', 'end_lowest', 'end_flat', 'flat_middle', 'end_slight', 'decline_end', 'end_stay', 'drop_end', 'increase_steady', 'middle_point', 'end_point', 'flat_remain', 'end_steady', 'middle_onward', 'increase_steadily', 'decrease_steady', 'bottom_middle', 'end_rise', 'maximum_value', 'end_sharp', 'highest_point', 'line_stay', 'line_stagnant', 'increase_throughout', 'beginning_value', 'lower_start', 'beginning_end', 'start_value', 'beginning_sharply', 'end_higher', 'increase_slight', 'increase_start', 'near_peak', 'decline_middle']
# ---- manually chosen:
# [ 'beginning_increase', 'increase_middle',  'end_increase',  'beginning_decrease', 'decrease_middle', 'decrease_end', 'flat_stay',  'middle_peak', 'end_peak'] ::  Counter({0: 399, 1: 164, 2: 7}) # 31%
# ,'flat_middle','increase_throughout'



# env PYTHONPATH=. python allennlp_series/data/dataset_reader/stock_text_reader.py

# metrics =  {'Bleu_1': 0.2795358649786081, 'Bleu_2': 0.1717628189124877, 'Bleu_3': 0.09778243075719946, 'Bleu_4': 0.05600989009969427, 'METEOR': 0.11977978413767922, 'ROUGE_L': 0.2570338104164155, 'CIDEr': 0.18586866874941843}



