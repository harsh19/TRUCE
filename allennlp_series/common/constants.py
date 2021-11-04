SERIES_TYPE_SINGLE = "single"
SERIES_TYPE_MULTI = "multi"
SERIES_TYPES_LIST = [SERIES_TYPE_SINGLE, SERIES_TYPE_MULTI]


# MODEL TASK SETTINGS
BINARY_CLASSIFICATION_SETTING = "binary_classification"

#
POSITIVE_LABEL_TMP = "positive"
NEGATIVE_LABEL_TMP = "negative"
STRLABEL_TO_INT = {POSITIVE_LABEL_TMP: "1", NEGATIVE_LABEL_TMP: "0"}

LABEL_DATA_TYPE_CAT = "categorical"
LABEL_DATA_TYPE_TEXT1 = "text1"
LABEL_DATA_TYPE_TEXT2 = "text2"
LABEL_DATA_TYPE_SUBSET = "subset"
LABEL_DATA_TYPE_LIST = [
    LABEL_DATA_TYPE_TEXT1,
    LABEL_DATA_TYPE_TEXT2,
    LABEL_DATA_TYPE_CAT,
    LABEL_DATA_TYPE_SUBSET,
]

BASE_ADJUST = 0.3775
BASE_SCALE = 10

LOCATE_SZ=12
LOCATE_KERNEL_STD = 0.1
PATT_MODULE_LAYER1_BIAS = 1.5

LABEL_DATA_TYPE_SUBSET_TYPE0 = [1, 2]  # increase begin and increase mid
LABEL_DATA_TYPE_SUBSET_TYPE1 = [0, 1]  # increase all and increase begin
LABEL_DATA_TYPE_SUBSET_TYPE2 = [
    1,
    2,
    6,
    7,
]  # increase beginning, inc-mid, dec-mid, decrease end
LABEL_DATA_TYPE_SUBSET_TYPE2_COMPLIMENT = [
    3,
    5,
]  # increase beginning, inc-mid, dec-mid, decrease end
LABEL_DATA_TYPE_SUBSET_TYPE2b = [
    1,
    2,
    3,
    6,
    7,
]  # increase beginning, inc-mid, inc-end, dec-mid, decrease end
LABEL_DATA_TYPE_SUBSET_TYPE2b_COMPLIMENT = [
    5
]  # increase beginning, inc-mid, inc-end, dec-mid, decrease end
LABEL_DATA_TYPE_SUBSET_TYPE3 = [1, 2, 3, 5, 6, 7]
LABEL_DATA_TYPE_SUBSET_TYPE4 = [1, 2, 6, 7, 8, 9, 12, 13]
LABEL_DATA_TYPE_SUBSET_TYPE5 = [1]  # oncrease begin
LABEL_DATA_TYPE_SUBSET_TYPE5b = [1, 2, 3]  # increase negin, mid, end
LABEL_DATA_TYPE_SUBSET_TYPE5c = [1, 2, 3, 4]  # increase all
LABEL_DATA_TYPE_SUBSET_TYPE6 = [8, 10, 11, 13]  # peak and trough ; begin and end
LABEL_DATA_TYPE_SUBSET_TYPE6b = [
    8,
    9,
    10,
    11,
    12,
    13,
]  # peak and trough; begin,middle,end
LABEL_DATA_TYPE_SUBSET_TYPEallbut = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]
LABEL_DATA_TYPE_SUBSET_TYPEall = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
LABEL_DATA_TYPE_SUBSET_TYPE_LIST = {
    "type0": LABEL_DATA_TYPE_SUBSET_TYPE0,
    "type1": LABEL_DATA_TYPE_SUBSET_TYPE1,
    "type2": LABEL_DATA_TYPE_SUBSET_TYPE2,
    "type2_complement": LABEL_DATA_TYPE_SUBSET_TYPE2_COMPLIMENT,
    "type2b": LABEL_DATA_TYPE_SUBSET_TYPE2b,
    "type2b_complement": LABEL_DATA_TYPE_SUBSET_TYPE2b_COMPLIMENT,
    "type3": LABEL_DATA_TYPE_SUBSET_TYPE3,
    "type4": LABEL_DATA_TYPE_SUBSET_TYPE4,
    "type5": LABEL_DATA_TYPE_SUBSET_TYPE5,
    "type5b": LABEL_DATA_TYPE_SUBSET_TYPE5b,
    "type5c": LABEL_DATA_TYPE_SUBSET_TYPE5c,
    "type6": LABEL_DATA_TYPE_SUBSET_TYPE6,
    "type6b": LABEL_DATA_TYPE_SUBSET_TYPE6b,
    "all_but_throughout": LABEL_DATA_TYPE_SUBSET_TYPEallbut,
    "all": LABEL_DATA_TYPE_SUBSET_TYPEall,
}

START_SYMBOL = "@START@"
END_SYMBOL = "@END@"


label_mapper = {
    0: "increase_all",
    1: "increase_begin",
    2: "increase_middle",
    3: "increase_end",
    4: "decrease_all",
    5: "decrease_begin",
    6: "decrease_middle",
    7: "decrease_end",
    8: "peak_begin",
    9: "peak_middle",
    10: "peak_end",
    11: "trough_begin",
    12: "trough_middle",
    13: "trough_end",
}

# programs: 0.inc-begin; 1.dec-begin; 2.inc-mid; 3.dec-mid; ...
# programs: inc-begin; dec-begin; peak-begin; trough-begin; ...

LABEL_TO_PROGRAM_MAPPER_6programs = {1: 0, 2: 2, 3: 4, 5: 1, 6: 3, 7: 5}

LABEL_TO_PROGRAM_MAPPER_12programs = {
    1: 0,
    2: 4,
    3: 8,
    5: 1,
    6: 5,
    7: 9,
    8: 2,
    9: 6,
    10: 10,
    11: 3,
    12: 7,
    13: 11,
}


word_pair_labels = [
    "none",
    "beginning_increase",
    "increase_middle",
    "end_increase",
    "beginning_decrease",
    "decrease_middle",
    "decrease_end",
    "flat_stay",
    "middle_peak",
    "end_peak",
]
word_pair_labels_factorized = {
    "default": ["none"],
    "locate": ["beginning", "middle", "end", "throughout"],
    "pattern": ["increase", "decrease", "peak", "flat"],
}
word_pair_labels_factorizedv2 = {
    "default": ["none"],
    "locate": ["beginning", "middle", "end", "throughout"],
    "pattern": ["increase", "decrease", "peak", "flat", "dip"],
}

word_pair_labels_v2 = [
    "none",
    "increase_middle",
    "decrease_end",
    "end_value",
    "end_increase",
    "middle_peak",
    "end_start",
    "end_toward",
    "beginning_increase",
    "flat_stay",
    "middle_rise",
    "end_near",
    "end_peak",
    "decrease_middle",
    "beginning_decrease",
    "end_lower",
    "lowest_point",
    "end_slightly",
    "end_lowest",
    "end_flat",
    "flat_middle",
    "end_slight",
    "decline_end",
    "end_stay",
    "drop_end",
]