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
import string
from allennlp_series.common.constants import *
import copy
from allennlp.data.tokenizers.word_stemmer import PorterStemmer, PassThroughWordStemmer
from allennlp_series.training.metrics import CocovalsMeasures
import random
from collections import Counter

import sys

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel("INFO")


def normalize_text(s):
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return s.strip()


def _24to12(val):
    return [v for i, v in enumerate(val) if i % 2 == 0]


@DatasetReader.register("synthetic_series_text_reader")
class SyntheticDataTextReader(DatasetReader):
    def __init__(
        self,
        # tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_pieces: int = 512,
        debug: bool = False,
        train_data_fraction: float = 1.0,
        perform_stemming: bool = False,
        label_data_subset_type: str = None,
        overfit_mode: bool = False,
        single_word_label_mode: bool = False,
        single_word_label_type: str = "one",
        self_eval_mode: bool = False,
        transformation: str = None,
        test_on_stock: bool = False,
        use_provided_vocab: str = None,  # used for loading inference n/w vocab for labels
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
        self.label_data_subset_type = label_data_subset_type
        if label_data_subset_type is not None:
            assert label_data_subset_type in LABEL_DATA_TYPE_SUBSET_TYPE_LIST.keys()
        self.overfit_mode = overfit_mode
        self.file_record = {}
        self.single_word_label_mode = single_word_label_mode
        self.self_eval_mode = self_eval_mode
        if self.self_eval_mode:
            self.ngram_overlap_eval = CocovalsMeasures()
        self.single_word_label_type = single_word_label_type
        assert single_word_label_type in ["one", "two", "two_reversed"]
        self.transformation = transformation
        self._test_on_stock = test_on_stock
        self.label_vocab_provided = None
        if use_provided_vocab:
            self.label_vocab_provided = {
                label.strip(): idx
                for idx, label in enumerate(open(use_provided_vocab, "r").readlines())
            }
            # print("** self.label_vocab_provided = ", self.label_vocab_provided)
            logger.info(
                f"[SYNTH-TEXT-READER] self.label_vocab_provided = {self.label_vocab_provided}"
            )
            for idx, label in enumerate(open(use_provided_vocab, "r").readlines()):
                LabelField(label, skip_indexing=False)

    @overrides
    def _read(self, file_path: str):  # processed_data/pilot12val.json

        ##############################################
        logger.info("")
        logger.info("=" * 83)

        data_label_dist = {}

        if file_path.count("train") > 0:
            assert "train" not in self.file_record
            self.file_record["train"] = file_path
        elif file_path.count("dev") > 0:
            assert "dev" not in self.file_record
            self.file_record["dev"] = file_path

        if self.overfit_mode:  # for sanity checking the model capacity
            if file_path.count("dev") > 0 or file_path.count("val") > 0:  # dev split
                file_path = self.file_record["train"]
                logger.info("#" * 22)
                logger.info(f" FORCE CHANGING DEV FILE TO = {file_path}")
                logger.info("#" * 22)

        self._instance_num = 0
        skip_cnt = 0
        logger.info(
            f"[SYNTH-TEXT-READER] ---->>>>>>>>>>>>>>>>>>>>>>> Reading instances from file at: {file_path}"
        )
        data = json.load(open(file_path, "r"))

        sz = len(data)  # number of ids
        all_ids = list(data.keys())
        logger.info(f"sz : len(data) = {sz}")

        if file_path.count("train") > 0 and self._train_data_fraction < 1.0:
            new_sz = int(sz * self._train_data_fraction)
            all_ids = all_ids[:new_sz]
        logger.info(
            f"train_data_fraction: = {self._train_data_fraction} ;; len(all_ids) =  {len(all_ids)} "
        )
        # print("train_data_fraction: ", self._train_data_fraction, " len(image_ids) = ", len(image_ids))

        ##############################################

        inst_num = 0

        for i, idx in enumerate(all_ids):

            if self.debug_mode:
                pass
                # print("*********>>>>>>> img_id = ", img_id)

            row = data[idx]
            series = row["series"]
            if self.transformation is not None:
                if self.transformation == "24to12":
                    series = _24to12(series)
                else:
                    raise NotImplementedError
            idx = row["idx"]
            meta = row["meta"]
            meta.update({"idx": idx})  # data collection idx is the reference idx now on

            label_data = meta["label"]["labels"]
            # print("label_data = ", label_data)
            if self.label_data_subset_type is not None:
                label_data_subset = LABEL_DATA_TYPE_SUBSET_TYPE_LIST[
                    self.label_data_subset_type
                ]
                if label_data not in label_data_subset:
                    skip_cnt += len(row["annotations"])
                    continue

            annotations = row["annotations"]
            for j, annotation in enumerate(annotations):
                meta = copy.deepcopy(meta)
                meta["raw_text_used"] = meta["raw_text"] = annotation
                if self._test_on_stock:
                    txt = annotation
                else:
                    txt = annotation[0]
                if self.single_word_label_mode:
                    txt = label_mapper[label_data]
                    if self.single_word_label_type == "two":
                        txt = " ".join(txt.split("_"))
                    elif self.single_word_label_type == "two_reversed":
                        txt = " ".join(list(reversed(txt.split("_"))))
                    meta["raw_text_used"] = [txt]
                if self.overfit_mode and data_label_dist.get(label_data, 0) > 0:
                    continue
                data_label_dist[label_data] = data_label_dist.get(label_data, 0) + 1
                yield self.text_to_instance(series, txt, meta)
                inst_num += 1
                if self.debug_mode:
                    break

            if self.self_eval_mode:
                annotations_id = copy.deepcopy(row["annotations"])
                random.shuffle(annotations_id)
                predicted_str = annotations_id[0]
                for target_str in annotations_id[1:]:
                    self.ngram_overlap_eval(predicted_str[0], target_str[0], idx)
                    # self.ngram_overlap_eval(predicted_str[0], predicted_str[0], img_id) # Sanity Check
                    # break # with only 1 reference

            if self.debug_mode and inst_num > 1:
                break

        logger.info(
            f"[READER]: file_path =  {file_path} || skip_cnt = {skip_cnt} || inst_num = {inst_num}"
        )
        logger.info(
            f"[READER]: file_path = {file_path} ||| data_label_dist : {data_label_dist}"
        )

    @overrides
    def text_to_instance(self, series, label_text, meta=None) -> Instance:
        fields = {}
        series = np.array(series)
        fields["series"] = ArrayField(series)
        label_text = normalize_text(label_text)
        label_text = self._tokenizer.tokenize(label_text)
        fields["label_text"] = TextField(label_text, self._token_indexers)

        label_data = label_mapper[meta["label"]["labels"]]  # new
        # *** label id to label: _index_to_token [labels] {0: 'decrease_begin', 1: 'increase_middle', 2: 'increase_end', 3: 'decrease_middle', 4: 'decrease_end', 5: 'increase_begin'}
        if False:  # self.label_vocab_provided:
            fields["label"] = LabelField(
                self.label_vocab_provided[label_data], skip_indexing=True
            )  # new
            print("-->> fields['label'] = ", fields["label"])
        else:
            fields["label"] = LabelField(label_data, skip_indexing=False)  # new
            logger.debug(f"-->> fields['label'] = {fields['label']} ")

        if self.debug_mode:
            logger.debug(f"meta =  {meta} ")
        metadata = meta
        fields["metadata"] = MetadataField(metadata)
        ins = Instance(fields=fields)
        logger.debug(f"=====>>>> ins = {ins} ")

        return ins


if __name__ == "__main__":

    # reader = SyntheticDataReader(tokenizer=None,
    #                              debug=True)
    # cnt = 0
    # for ins in reader._read("../data_collection/tables/synthetic_data/data1_10k_12r_dev.pkl"):
    #     print("ins = ", ins)
    #     print("cnt = ", cnt)

    #
    # reader = SyntheticDataTextReader(debug=False,
    #                         label_data_subset_type='all', #''all_but_throughout',
    #                         perform_stemming = False,
    #                         add_distractors_data = False,
    #                         self_eval_mode=True)

    random.seed(123)
    np.random.seed(123)

    reader = SyntheticDataTextReader(
        debug=False,
        label_data_subset_type="type3",  #''all_but_throughout',
        perform_stemming=False,
        self_eval_mode=True,
    )
    # there is an option for Class-wise random when doing self_eval. has to be manually adjusted

    cnt = 0
    # for ins in reader._read("processed_data/pilot13val.json"):
    for ins in reader._read("processed_data/pilot13finalval.json"):
        print("cnt = ", cnt)
        print("ins = ", ins)
        print("ins: meta = ", ins.fields["metadata"])
        cnt += 1
        print("=" * 31)

    metrics = reader.ngram_overlap_eval.get_metric(reset=True)
    print("metrics = ", metrics)

    # ****** Random Prediction
    # reader._self_eval_random("processed_data/pilot13val.json")
    # metrics = reader.ngram_overlap_eval.get_metric(reset=True)
    # print("metrics = ", metrics)

    # reader._self_eval_random("processed_data/pilot13finalval.json")

    # ****** stats
    vals = {}
    all_tokens = []
    import scipy.stats

    for split in ["train", "val"]:
        inst_cnt = 0
        ctr = {}
        lengths = []
        # for inst in reader.read("processed_data/pilot13h"+split+".json"):
        for inst in reader.read("processed_data/pilot13final" + split + ".json"):
            logger.debug(f"inst['metadata'].metadata =  {inst['metadata'].metadata}")
            label = inst["metadata"].metadata["label"]["labels"]
            logger.debug(f"label = {label}")
            ctr[label] = ctr.get(label, 0) + 1
            logger.debug(f"inst['label_text'].tokens = {inst['label_text'].tokens}")
            lengths.append(len(inst["label_text"].tokens) - 2)
            all_tokens.extend(inst["label_text"].tokens[1:-1])
            inst_cnt += 1
        logger.debug("==========")
        logger.debug(f"inst_cnt = {inst_cnt}")
        vals[split] = {}
        vals[split]["inst_cnt"] = inst_cnt
        vals[split]["ctr"] = ctr
        vals[split]["lengths"] = Counter(lengths)
        vals[split]["lengths-summary"] = scipy.stats.describe(lengths)
    print("---------------")
    print("vals = ", vals)
    print("=" * 21)
    print("-----VOCABULARY------")
    all_tokens = [str(t) for t in all_tokens]
    print("#count = ", len(all_tokens))
    dct_tokens = dict(Counter(all_tokens))
    print("#SortedByCounts = ", sorted(dct_tokens.items(), key=lambda x: x[1]))
    print("#uniq-count = ", len(dct_tokens))

    # fw = open('synth_data_vocab.tsv','w')
    # for kv in  sorted( list(dict(Counter(all_tokens)).items()), key=lambda x:x[1]):
    #     k,v = kv
    #     fw.write('\t'.join([str(k),str(v)]))
    #     fw.write('\n')
    # fw.close()

    # inst_cnt = 72
    # ---------------
    # vals = {'train': {'inst_cnt': 576, 'ctr': {2: 96, 7: 96, 1: 96, 3: 96, 6: 96, 5: 96},
    #                   'lengths': Counter({4: 215, 5: 211, 6: 56, 7: 30, 3: 28, 8: 24, 2: 10, 0: 1, 1: 1}),
    #                   'lengths-summary': DescribeResult(nobs=576, minmax=(0, 8), mean=4.788194444444445,
    #                                                     variance=1.4333212560386472, skewness=0.5834199563557133,
    #                                                     kurtosis=1.323387181466785)},
    #         'val': {'inst_cnt': 72, 'ctr': {3: 12, 7: 12, 2: 12, 6: 12, 1: 12, 5: 12},
    #                 'lengths': Counter({4: 31, 5: 26, 6: 7, 8: 3, 3: 3, 7: 2}),
    #                 'lengths-summary': DescribeResult(nobs=72, minmax=(3, 8), mean=4.763888888888889,
    #                                                   variance=1.1406494522691706, skewness=1.2494023100222054,
    #                                                   kurtosis=1.7757192300444657)}}
    # == == == == == == == == == == =
    # -----VOCABULARY - -----
    # # count =  3101

    #

