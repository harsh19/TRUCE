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
from allennlp.data.tokenizers import Token, WordTokenizer #, WhitespaceTokenizer
# from allennlp_series.data.analysis.entity_detection import EntityDetectionOverlap, EntityDetectionAll
# from allennlp_series.training.metrics import F1MeasureCustom, ExplanationEval, PrecisionEval
import string
from allennlp_series.common.constants import *
import copy
from allennlp.data.tokenizers.word_stemmer import PorterStemmer , PassThroughWordStemmer
from allennlp_series.training.metrics import CocovalsMeasures
import random
from collections import Counter
from allennlp.data.tokenizers import Tokenizer

logger = logging.getLogger(__name__)

def normalize_text(s):
    s = s.lower().strip()
    s = s.translate(str.maketrans('', '', string.punctuation))
    # s = START_SYMBOL + ' ' + s  + ' ' +  END_SYMBOL
    return s.strip()

stock_names = ['aapl','apple','googl','google','msft','microsoft','amzn','amazon']

'''
{"https://github.com/poetry-gen-samples/tables/raw/master/outputs/pilot12/3.png": 
    {"id": "https://github.com/poetry-gen-samples/tables/raw/master/outputs/pilot12/3.png", 
    "idx": "3", # this is idx at time of image creation (gets shuffled from original dump) **
    # the obj in the folder corresponds to this idx
    "annotations": [
        ["Google is taking a very sharp decline."], 
        ["GOOGL declines sharply at the beginning"], 
        ["GOOGL in continual, even decline throughout"]], 
    "series": [51, 47, 43, 39, 35, 31, 27, 23, 19, 15, 11, 7], 
    "meta": {
        "col_names": "GOOGL", 
        "label": {
            "selector": 4, 
            "labels": 4, 
            "start_point": -1, 
            "end_point": -1, 
            "stretch": -1}, 
        "idx": 5348} # ** this idx is idx from original dump 
    }
'''

# import allennlp_series.common.constants
# label_mapper = constants.label_mapper



@DatasetReader.register("inference_reader")
class InferenceReader(DatasetReader):

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_pieces: int = 512,
                 debug: bool = False,
                 train_data_fraction:float = 1.0,
                 perform_stemming: bool = False,
                 label_data_subset_type: str = None,
                 overfit_mode: bool = False,
                 single_word_label_mode: bool = False,
                 single_word_label_type: str = 'one',
                 self_eval_mode: bool = False,
                 add_distractors_data: bool = False,
                 mode_type : str = 'normal_mode',
                 lazy: bool = False,
                 label_type: str = 'complete',
                 bert_model = False) -> None:
        super().__init__(lazy)
        self._perform_stemming = perform_stemming
        if perform_stemming:
            self._stemmer = PorterStemmer()
        else:
            self._stemmer = PassThroughWordStemmer()
        if not bert_model:
            self._tokenizer =  WordTokenizer( start_tokens=[START_SYMBOL],
                                          end_tokens=[END_SYMBOL],
                                          word_stemmer=self._stemmer
                                        )
            self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        else:
            self._tokenizer = tokenizer
            self._token_indexers = token_indexers
        self._max_pieces = max_pieces
        self.debug_mode = debug
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
        self.add_distractors_data = add_distractors_data
        if add_distractors_data:
            self.distractors_label_to_annotations = {}
        self.single_word_label_type = single_word_label_type
        assert single_word_label_type in ['one','two','two_reversed']
        self.mode_type = mode_type
        self.bert_model = bert_model
        assert label_type in ['complete','factorized','trend','locate']
        self.label_type = label_type


    @overrides
    def _read(self, file_path: str): # processed_data/pilot12val.json

        ##############################################
        print()
        print("="*83)

        data_label_dist = {}

        self._instance_num = 0
        skip_cnt = 0
        print("---->>>>>>>>>>>>>>>>>>>>>>> Reading instances from file at: ", file_path)
        mode_type = self.mode_type

        if mode_type == 'normal_mode':
            data = json.load( open(file_path,'r') )
        elif self.mode_type == 'transfer_mode':
            if file_path.count('pkl')>0:
                import pickle
                data = pickle.load(open(file_path, 'rb'))
            else:
                mode_type = 'normal_mode'
                data = json.load( open(file_path,'r') )
        else:
            raise NotImplementedError

        if mode_type == 'normal_mode':

            sz = len(data) # number of images
            image_ids = list(data.keys())
            print("sz : len(image_ids) = ", sz)

            if file_path.count('train')>0 and self._train_data_fraction < 1.0:
                new_sz = int(sz*self._train_data_fraction)
                image_ids = image_ids[:new_sz]
            print("train_data_fraction: ", self._train_data_fraction, " len(image_ids) = ", len(image_ids))


            ##############################################

            inst_num = 0

            for i,img_id in enumerate(image_ids):

                row = data[img_id]
                series = row['series']
                idx = row['idx']
                meta = row['meta']
                meta.update({'idx':idx}) # data collection idx is the reference idx now on

                if 'label' in meta:
                    label_data = meta['label']['labels']
                else:
                    label_data = None
                #print("label_data = ", label_data)
                if self.label_data_subset_type is not None:
                    label_data_subset = LABEL_DATA_TYPE_SUBSET_TYPE_LIST[self.label_data_subset_type]
                    if label_data not in label_data_subset:
                        skip_cnt += len(row['annotations'])
                        continue

                print("meta = ", meta)
                if label_data:
                    label_data = label_mapper[label_data]
                    if self.label_type == 'complete':
                        pass
                    elif self.label_type == 'trend':
                        label_data = label_data.split('_')[0]
                    elif self.label_type == 'locate':
                        label_data = label_data.split('_')[1]
                    elif self.label_type == 'factorized':
                        label_data = label_data.split('_')
                    else:
                        raise NotImplementedError

                annotations = row['annotations']
                for j,annotation in enumerate(annotations):
                    meta = copy.deepcopy(meta)
                    meta['raw_text_used'] = meta['raw_text'] = annotation
                    txt = annotation[0]
                    if self.single_word_label_mode:
                        txt = label_data #label_mapper[label_data]
                        if self.label_type == 'factorized':
                            txt = '_'.join(label_data)
                        if self.single_word_label_type == 'two':
                            txt = ' '.join(txt.split('_'))
                        elif self.single_word_label_type == 'two_reversed':
                            txt = ' '.join(list(reversed(txt.split('_'))))
                        meta['raw_text_used'] = [txt]
                    if self.label_type == 'factorized':
                        label_data_str = '_'.join(label_data)
                        data_label_dist[label_data_str] = data_label_dist.get(label_data_str, 0) + 1
                    else:
                        if label_data:
                            data_label_dist[label_data] = data_label_dist.get(label_data, 0) + 1
                    yield self.text_to_instance(series, txt, label_data, meta)
                    inst_num += 1
                    if self.debug_mode:
                        break

                if self.debug_mode and inst_num>1:
                    break

            print("file_path = ", file_path, " || skip_cnt = ", skip_cnt, " || inst_num = ", inst_num)
            print("file_path = ", file_path, " ||| data_label_dist : ", data_label_dist)

        else:
            raise NotImplementedError
            inst_num = 0
            all_cols, all_labels = data
            label_data_subset_type = LABEL_DATA_TYPE_SUBSET_TYPE_LIST[self.label_data_subset_type]
            label_list = label_data_subset_type
            for col, label in zip(all_cols, all_labels):
                cur_label = label['labels']
                if cur_label not in label_list:
                    continue
                else:
                    cur_label = label['label_text']  # label_to_labeltext[cur_label]
                    print("[Reader] cur_label = ", cur_label)
                    yield self.text_to_instance(col, cur_label, cur_label)
                if self.debug_mode and inst_num >= 31:
                    break
                inst_num += 1


    @overrides
    def text_to_instance(self, series, label_text, label_data, meta=None) -> Instance:
        fields = {}

        label_text = normalize_text(label_text)
        # label_text = self._tokenizer.split_words(label_text)
        print("self._tokenizer = ", self._tokenizer)
        label_text = self._tokenizer.tokenize(label_text)

        # label_text = [t if t.text not in stock_names else Token('series') for t in label_text ] -- no longer needed

        fields['tokens'] = TextField(label_text, self._token_indexers)
        if label_data:
            if self.label_type == 'factorized':
                fields['label'] = LabelField(label_data[0], skip_indexing=False)
                fields['label2'] = LabelField(label_data[1], skip_indexing=False, label_namespace='2labels')
            else:
                fields['label'] = LabelField(label_data, skip_indexing=False)

        if self.debug_mode:
            print("meta = ", meta)
        metadata = meta

        if not self.bert_model:
            series = np.array(series)
            fields['series'] = ArrayField(series)
            fields["metadata"] = MetadataField(metadata)

        ins = Instance(fields=fields)
        print("=====>>>> ins = ", ins)

        return ins





if __name__ == "__main__":

    reader = InferenceReader(debug=False,
                            label_data_subset_type='all_but_throughout',
                            perform_stemming = True)

    # reader._self_eval_random("processed_data/pilot13val.json")
    # metrics = reader.ngram_overlap_eval.get_metric(reset=True)
    # print("metrics = ", metrics)

    for ins in reader._read("processed_data/pilot13val.json"):
        print(ins)







