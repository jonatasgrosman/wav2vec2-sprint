# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Common Voice Dataset"""

from __future__ import absolute_import, division, print_function

import os
import re
import homoglyphs as hg
import gdown
import json
import pandas as pd
import glob

import datasets

import soundfile as sf
import librosa
import warnings

from lang_trans.arabic import buckwalter

_DATA_URL = "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-6.1-2020-12-11/{}.tar.gz"

_CITATION = """\
@inproceedings{commonvoice:2020,
  author = {Ardila, R. and Branson, M. and Davis, K. and Henretty, M. and Kohler, M. and Meyer, J. and Morais, R. and Saunders, L. and Tyers, F. M. and Weber, G.},
  title = {Common Voice: A Massively-Multilingual Speech Corpus},
  booktitle = {Proceedings of the 12th Conference on Language Resources and Evaluation (LREC 2020)},
  pages = {4211--4215},
  year = 2020
}
"""

_DESCRIPTION = """\
Common Voice is Mozilla's initiative to help teach machines how real people speak.
The dataset currently consists of 7,335 validated hours of speech in 60 languages, but we’re always adding more voices and languages.
"""

_HOMEPAGE = "https://commonvoice.mozilla.org/en/datasets"

_LICENSE = "https://github.com/common-voice/common-voice/blob/main/LICENSE"

_LANGUAGES = {
    "ab": {
        "Language": "Abkhaz",
        "Date": "2020-12-11",
        "Size": "39 MB",
        "Version": "ab_1h_2020-12-11",
        "Validated_Hr_Total": 0.05,
        "Overall_Hr_Total": 1,
        "Number_Of_Voice": 14,
    },
    "ar": {
        "Language": "Arabic",
        "Date": "2020-12-11",
        "Size": "2 GB",
        "Version": "ar_77h_2020-12-11",
        "Validated_Hr_Total": 49,
        "Overall_Hr_Total": 77,
        "Number_Of_Voice": 672,
    },
    "as": {
        "Language": "Assamese",
        "Date": "2020-12-11",
        "Size": "21 MB",
        "Version": "as_0.78h_2020-12-11",
        "Validated_Hr_Total": 0.74,
        "Overall_Hr_Total": 0.78,
        "Number_Of_Voice": 17,
    },
    "br": {
        "Language": "Breton",
        "Date": "2020-12-11",
        "Size": "444 MB",
        "Version": "br_16h_2020-12-11",
        "Validated_Hr_Total": 7,
        "Overall_Hr_Total": 16,
        "Number_Of_Voice": 157,
    },
    "ca": {
        "Language": "Catalan",
        "Date": "2020-12-11",
        "Size": "19 GB",
        "Version": "ca_748h_2020-12-11",
        "Validated_Hr_Total": 623,
        "Overall_Hr_Total": 748,
        "Number_Of_Voice": 5376,
    },
    "cnh": {
        "Language": "Hakha Chin",
        "Date": "2020-12-11",
        "Size": "39 MB",
        "Version": "ab_1h_2020-12-11",
        "Validated_Hr_Total": 0.05,
        "Overall_Hr_Total": 1,
        "Number_Of_Voice": 14,
    },
    "cs": {
        "Language": "Czech",
        "Date": "2020-12-11",
        "Size": "39 MB",
        "Version": "ab_1h_2020-12-11",
        "Validated_Hr_Total": 0.05,
        "Overall_Hr_Total": 1,
        "Number_Of_Voice": 14,
    },
    "cv": {
        "Language": "Chuvash",
        "Date": "2020-12-11",
        "Size": "419 MB",
        "Version": "cv_16h_2020-12-11",
        "Validated_Hr_Total": 4,
        "Overall_Hr_Total": 16,
        "Number_Of_Voice": 92,
    },
    "cy": {
        "Language": "Welsh",
        "Date": "2020-12-11",
        "Size": "3 GB",
        "Version": "cy_124h_2020-12-11",
        "Validated_Hr_Total": 95,
        "Overall_Hr_Total": 124,
        "Number_Of_Voice": 1382,
    },
    "de": {
        "Language": "German",
        "Date": "2020-12-11",
        "Size": "22 GB",
        "Version": "de_836h_2020-12-11",
        "Validated_Hr_Total": 777,
        "Overall_Hr_Total": 836,
        "Number_Of_Voice": 12659,
    },
    "dv": {
        "Language": "Dhivehi",
        "Date": "2020-12-11",
        "Size": "515 MB",
        "Version": "dv_19h_2020-12-11",
        "Validated_Hr_Total": 18,
        "Overall_Hr_Total": 19,
        "Number_Of_Voice": 167,
    },
    "el": {
        "Language": "Greek",
        "Date": "2020-12-11",
        "Size": "364 MB",
        "Version": "el_13h_2020-12-11",
        "Validated_Hr_Total": 6,
        "Overall_Hr_Total": 13,
        "Number_Of_Voice": 118,
    },
    "en": {
        "Language": "English",
        "Date": "2020-12-11",
        "Size": "56 GB",
        "Version": "en_2181h_2020-12-11",
        "Validated_Hr_Total": 1686,
        "Overall_Hr_Total": 2181,
        "Number_Of_Voice": 66173,
    },
    "eo": {
        "Language": "Esperanto",
        "Date": "2020-12-11",
        "Size": "3 GB",
        "Version": "eo_102h_2020-12-11",
        "Validated_Hr_Total": 90,
        "Overall_Hr_Total": 102,
        "Number_Of_Voice": 574,
    },
    "es": {
        "Language": "Spanish",
        "Date": "2020-12-11",
        "Size": "15 GB",
        "Version": "es_579h_2020-12-11",
        "Validated_Hr_Total": 324,
        "Overall_Hr_Total": 579,
        "Number_Of_Voice": 19484,
    },
    "et": {
        "Language": "Estonian",
        "Date": "2020-12-11",
        "Size": "732 MB",
        "Version": "et_27h_2020-12-11",
        "Validated_Hr_Total": 19,
        "Overall_Hr_Total": 27,
        "Number_Of_Voice": 543,
    },
    "eu": {
        "Language": "Basque",
        "Date": "2020-12-11",
        "Size": "3 GB",
        "Version": "eu_131h_2020-12-11",
        "Validated_Hr_Total": 89,
        "Overall_Hr_Total": 131,
        "Number_Of_Voice": 1028,
    },
    "fa": {
        "Language": "Persian",
        "Date": "2020-12-11",
        "Size": "8 GB",
        "Version": "fa_321h_2020-12-11",
        "Validated_Hr_Total": 282,
        "Overall_Hr_Total": 321,
        "Number_Of_Voice": 3655,
    },
    "fi": {
        "Language": "Finnish",
        "Date": "2020-12-11",
        "Size": "48 MB",
        "Version": "fi_1h_2020-12-11",
        "Validated_Hr_Total": 1,
        "Overall_Hr_Total": 1,
        "Number_Of_Voice": 27,
    },
    "fr": {
        "Language": "French",
        "Date": "2020-12-11",
        "Size": "18 GB",
        "Version": "fr_682h_2020-12-11",
        "Validated_Hr_Total": 623,
        "Overall_Hr_Total": 682,
        "Number_Of_Voice": 12953,
    },
    "fy-NL": {
        "Language": "Frisian",
        "Date": "2020-12-11",
        "Size": "1 GB",
        "Version": "fy-NL_46h_2020-12-11",
        "Validated_Hr_Total": 14,
        "Overall_Hr_Total": 46,
        "Number_Of_Voice": 467,
    },
    "ga-IE": {
        "Language": "Irish",
        "Date": "2020-12-11",
        "Size": "149 MB",
        "Version": "ga-IE_5h_2020-12-11",
        "Validated_Hr_Total": 3,
        "Overall_Hr_Total": 5,
        "Number_Of_Voice": 101,
    },
    "hi": {
        "Language": "Hindi",
        "Date": "2020-12-11",
        "Size": "20 MB",
        "Version": "hi_0.8h_2020-12-11",
        "Validated_Hr_Total": 0.54,
        "Overall_Hr_Total": 0.8,
        "Number_Of_Voice": 31,
    },
    "hsb": {
        "Language": "Sorbian, Upper",
        "Date": "2020-12-11",
        "Size": "76 MB",
        "Version": "hsb_2h_2020-12-11",
        "Validated_Hr_Total": 2,
        "Overall_Hr_Total": 2,
        "Number_Of_Voice": 19,
    },
    "hu": {
        "Language": "Hungarian",
        "Date": "2020-12-11",
        "Size": "232 MB",
        "Version": "hu_8h_2020-12-11",
        "Validated_Hr_Total": 8,
        "Overall_Hr_Total": 8,
        "Number_Of_Voice": 47,
    },
    "ia": {
        "Language": "InterLinguia",
        "Date": "2020-12-11",
        "Size": "216 MB",
        "Version": "ia_8h_2020-12-11",
        "Validated_Hr_Total": 6,
        "Overall_Hr_Total": 8,
        "Number_Of_Voice": 36,
    },
    "id": {
        "Language": "Indonesian",
        "Date": "2020-12-11",
        "Size": "454 MB",
        "Version": "id_17h_2020-12-11",
        "Validated_Hr_Total": 9,
        "Overall_Hr_Total": 17,
        "Number_Of_Voice": 219,
    },
    "it": {
        "Language": "Italian",
        "Date": "2020-12-11",
        "Size": "5 GB",
        "Version": "it_199h_2020-12-11",
        "Validated_Hr_Total": 158,
        "Overall_Hr_Total": 199,
        "Number_Of_Voice": 5729,
    },
    "ja": {
        "Language": "Japanese",
        "Date": "2020-12-11",
        "Size": "146 MB",
        "Version": "ja_5h_2020-12-11",
        "Validated_Hr_Total": 3,
        "Overall_Hr_Total": 5,
        "Number_Of_Voice": 235,
    },
    "ka": {
        "Language": "Georgian",
        "Date": "2020-12-11",
        "Size": "99 MB",
        "Version": "ka_3h_2020-12-11",
        "Validated_Hr_Total": 3,
        "Overall_Hr_Total": 3,
        "Number_Of_Voice": 44,
    },
    "kab": {
        "Language": "Kabyle",
        "Date": "2020-12-11",
        "Size": "16 GB",
        "Version": "kab_622h_2020-12-11",
        "Validated_Hr_Total": 525,
        "Overall_Hr_Total": 622,
        "Number_Of_Voice": 1309,
    },
    "ky": {
        "Language": "Kyrgyz",
        "Date": "2020-12-11",
        "Size": "553 MB",
        "Version": "ky_22h_2020-12-11",
        "Validated_Hr_Total": 11,
        "Overall_Hr_Total": 22,
        "Number_Of_Voice": 134,
    },
    "lg": {
        "Language": "Luganda",
        "Date": "2020-12-11",
        "Size": "199 MB",
        "Version": "lg_8h_2020-12-11",
        "Validated_Hr_Total": 3,
        "Overall_Hr_Total": 8,
        "Number_Of_Voice": 76,
    },
    "lt": {
        "Language": "Lithuanian",
        "Date": "2020-12-11",
        "Size": "129 MB",
        "Version": "lt_4h_2020-12-11",
        "Validated_Hr_Total": 2,
        "Overall_Hr_Total": 4,
        "Number_Of_Voice": 30,
    },
    "lv": {
        "Language": "Latvian",
        "Date": "2020-12-11",
        "Size": "199 MB",
        "Version": "lv_7h_2020-12-11",
        "Validated_Hr_Total": 6,
        "Overall_Hr_Total": 7,
        "Number_Of_Voice": 99,
    },
    "mn": {
        "Language": "Mongolian",
        "Date": "2020-12-11",
        "Size": "464 MB",
        "Version": "mn_17h_2020-12-11",
        "Validated_Hr_Total": 11,
        "Overall_Hr_Total": 17,
        "Number_Of_Voice": 376,
    },
    "mt": {
        "Language": "Maltese",
        "Date": "2020-12-11",
        "Size": "405 MB",
        "Version": "mt_15h_2020-12-11",
        "Validated_Hr_Total": 7,
        "Overall_Hr_Total": 15,
        "Number_Of_Voice": 171,
    },
    "nl": {
        "Language": "Dutch",
        "Date": "2020-12-11",
        "Size": "2 GB",
        "Version": "nl_63h_2020-12-11",
        "Validated_Hr_Total": 59,
        "Overall_Hr_Total": 63,
        "Number_Of_Voice": 1012,
    },
    "or": {
        "Language": "Odia",
        "Date": "2020-12-11",
        "Size": "190 MB",
        "Version": "or_7h_2020-12-11",
        "Validated_Hr_Total": 0.87,
        "Overall_Hr_Total": 7,
        "Number_Of_Voice": 34,
    },
    "pa-IN": {
        "Language": "Punjabi",
        "Date": "2020-12-11",
        "Size": "67 MB",
        "Version": "pa-IN_2h_2020-12-11",
        "Validated_Hr_Total": 0.5,
        "Overall_Hr_Total": 2,
        "Number_Of_Voice": 26,
    },
    "pl": {
        "Language": "Polish",
        "Date": "2020-12-11",
        "Size": "3 GB",
        "Version": "pl_129h_2020-12-11",
        "Validated_Hr_Total": 108,
        "Overall_Hr_Total": 129,
        "Number_Of_Voice": 2647,
    },
    "pt": {
        "Language": "Portuguese",
        "Date": "2020-12-11",
        "Size": "2 GB",
        "Version": "pt_63h_2020-12-11",
        "Validated_Hr_Total": 50,
        "Overall_Hr_Total": 63,
        "Number_Of_Voice": 1120,
    },
    "rm-sursilv": {
        "Language": "Romansh Sursilvan",
        "Date": "2020-12-11",
        "Size": "263 MB",
        "Version": "rm-sursilv_9h_2020-12-11",
        "Validated_Hr_Total": 5,
        "Overall_Hr_Total": 9,
        "Number_Of_Voice": 78,
    },
    "rm-vallader": {
        "Language": "Romansh Vallader",
        "Date": "2020-12-11",
        "Size": "103 MB",
        "Version": "rm-vallader_3h_2020-12-11",
        "Validated_Hr_Total": 2,
        "Overall_Hr_Total": 3,
        "Number_Of_Voice": 39,
    },
    "ro": {
        "Language": "Romanian",
        "Date": "2020-12-11",
        "Size": "250 MB",
        "Version": "ro_9h_2020-12-11",
        "Validated_Hr_Total": 6,
        "Overall_Hr_Total": 9,
        "Number_Of_Voice": 130,
    },
    "ru": {
        "Language": "Russian",
        "Date": "2020-12-11",
        "Size": "3 GB",
        "Version": "ru_130h_2020-12-11",
        "Validated_Hr_Total": 111,
        "Overall_Hr_Total": 130,
        "Number_Of_Voice": 1412,
    },
    "rw": {
        "Language": "Kinyarwanda",
        "Date": "2020-12-11",
        "Size": "40 GB",
        "Version": "rw_1510h_2020-12-11",
        "Validated_Hr_Total": 1183,
        "Overall_Hr_Total": 1510,
        "Number_Of_Voice": 410,
    },
    "sah": {
        "Language": "Sakha",
        "Date": "2020-12-11",
        "Size": "173 MB",
        "Version": "sah_6h_2020-12-11",
        "Validated_Hr_Total": 4,
        "Overall_Hr_Total": 6,
        "Number_Of_Voice": 42,
    },
    "sl": {
        "Language": "Slovenian",
        "Date": "2020-12-11",
        "Size": "212 MB",
        "Version": "sl_7h_2020-12-11",
        "Validated_Hr_Total": 5,
        "Overall_Hr_Total": 7,
        "Number_Of_Voice": 82,
    },
    "sv-SE": {
        "Language": "Swedish",
        "Date": "2020-12-11",
        "Size": "402 MB",
        "Version": "sv-SE_15h_2020-12-11",
        "Validated_Hr_Total": 12,
        "Overall_Hr_Total": 15,
        "Number_Of_Voice": 222,
    },
    "ta": {
        "Language": "Tamil",
        "Date": "2020-12-11",
        "Size": "648 MB",
        "Version": "ta_24h_2020-12-11",
        "Validated_Hr_Total": 14,
        "Overall_Hr_Total": 24,
        "Number_Of_Voice": 266,
    },
    "th": {
        "Language": "Thai",
        "Date": "2020-12-11",
        "Size": "325 MB",
        "Version": "th_12h_2020-12-11",
        "Validated_Hr_Total": 8,
        "Overall_Hr_Total": 12,
        "Number_Of_Voice": 182,
    },
    "tr": {
        "Language": "Turkish",
        "Date": "2020-12-11",
        "Size": "592 MB",
        "Version": "tr_22h_2020-12-11",
        "Validated_Hr_Total": 20,
        "Overall_Hr_Total": 22,
        "Number_Of_Voice": 678,
    },
    "tt": {
        "Language": "Tatar",
        "Date": "2020-12-11",
        "Size": "741 MB",
        "Version": "tt_28h_2020-12-11",
        "Validated_Hr_Total": 26,
        "Overall_Hr_Total": 28,
        "Number_Of_Voice": 185,
    },
    "uk": {
        "Language": "Ukrainian",
        "Date": "2020-12-11",
        "Size": "1 GB",
        "Version": "uk_43h_2020-12-11",
        "Validated_Hr_Total": 30,
        "Overall_Hr_Total": 43,
        "Number_Of_Voice": 459,
    },
    "vi": {
        "Language": "Vietnamese",
        "Date": "2020-12-11",
        "Size": "50 MB",
        "Version": "vi_1h_2020-12-11",
        "Validated_Hr_Total": 0.74,
        "Overall_Hr_Total": 1,
        "Number_Of_Voice": 62,
    },
    "vot": {
        "Language": "Votic",
        "Date": "2020-12-11",
        "Size": "7 MB",
        "Version": "vot_0.28h_2020-12-11",
        "Validated_Hr_Total": 0,
        "Overall_Hr_Total": 0.28,
        "Number_Of_Voice": 3,
    },
    "zh-CN": {
        "Language": "Chinese (China)",
        "Date": "2020-12-11",
        "Size": "2 GB",
        "Version": "zh-CN_78h_2020-12-11",
        "Validated_Hr_Total": 56,
        "Overall_Hr_Total": 78,
        "Number_Of_Voice": 3501,
    },
    "zh-HK": {
        "Language": "Chinese (Hong Kong)",
        "Date": "2020-12-11",
        "Size": "3 GB",
        "Version": "zh-HK_100h_2020-12-11",
        "Validated_Hr_Total": 50,
        "Overall_Hr_Total": 100,
        "Number_Of_Voice": 2536,
    },
    "zh-TW": {
        "Language": "Chinese (Taiwan)",
        "Date": "2020-12-11",
        "Size": "2 GB",
        "Version": "zh-TW_78h_2020-12-11",
        "Validated_Hr_Total": 55,
        "Overall_Hr_Total": 78,
        "Number_Of_Voice": 1444,
    },
}

_CSS10_URLS = {
    "de": "https://drive.google.com/uc?id=1wgCHGvT0S8YrNfRTVyn23sW-5MFknoHA", # 7427 samples
    "el": "https://drive.google.com/uc?id=10BNORyOqkosxEf3qAAtWM1qWjHEZzXTO", # 1844 samples
    "es": "https://drive.google.com/uc?id=1dyUvSxv0KowTseI35dE8UXpVsYFhEpQV", # 11100 samples
    "fi": "https://drive.google.com/uc?id=1H4-eGIgf4aK_s14uo-srbKMENpysuV2u", # 4842 samples
    "fr": "https://drive.google.com/uc?id=1kuhoDjhA_Cij0SJuMI_4kneDTR_cqahS", # 8648 samples
    "hu": "https://drive.google.com/uc?id=1ms2INJ1e0ChU0TMzgDYLa8jtoTK2gkmE", # 4515 samples
    "ja": "https://drive.google.com/uc?id=1E4k8FduAk-_wy85AQrGakZBcw2hLhmU6", # 6841 samples
    "nl": "https://drive.google.com/uc?id=1ji8QD4lJzInz2vomGkMafRjpz3gGBYsf", # 6494 samples
    "ru": "https://drive.google.com/uc?id=1tx3dpO8SX8CriF0YsK8XeISZc9yGRody", # 9599 samples
    "zh-CN": "https://drive.google.com/uc?id=1hliY4KD_I8y4FQg5zta9IDGN0HRQLRiv", # 2971 samples
}

_JSUT_URLS = {
    "ja": "http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip" # 7696 samples
}

_NST_URLS = {
    "sv-SE": {
        "metadata": "https://www.nb.no/sbfil/talegjenkjenning/16kHz_2020/se_2020/ADB_SWE_0467.tar.gz",
        "files": "https://www.nb.no/sbfil/talegjenkjenning/16kHz_2020/se_2020/lydfiler_16_1.tar.gz", # ? samples
    }
}

_FREE_ST_URLS = {
    "zh-CN": "https://www.openslr.org/resources/38/ST-CMDS-20170001_1-OS.tar.gz", # 102600 samples
}

_ARABIC_SPEECH = {
    "ar": "http://en.arabicspeechcorpus.com/arabic-speech-corpus.zip" # 1913 samples
}

_TIMIT = {
    "en": "https://data.deepai.org/timit.zip" # 4620 samples
}

_LIBRISPEECH_DL_URL = "http://www.openslr.org/resources/12/"
_LIBRISPEECH = {
    "en": [
        _LIBRISPEECH_DL_URL + "dev-clean.tar.gz", # 2703 samples
        _LIBRISPEECH_DL_URL + "dev-other.tar.gz", # 2864 samples
        _LIBRISPEECH_DL_URL + "train-clean-100.tar.gz", # 28539 samples
        _LIBRISPEECH_DL_URL + "train-clean-360.tar.gz", # 104014 samples
        _LIBRISPEECH_DL_URL + "train-other-500.tar.gz", # 148688 samples
    ]
}

_MAX_TRAIN_SAMPLES = 90000
_MAX_VAL_SAMPLES = 10000

class CommonVoiceConfig(datasets.BuilderConfig):
    """BuilderConfig for CommonVoice."""

    def __init__(self, name, sub_version, **kwargs):
        """
        Args:
          data_dir: `string`, the path to the folder containing the files in the
            downloaded .tar
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          **kwargs: keyword arguments forwarded to super.
        """
        self.sub_version = sub_version
        self.language = kwargs.pop("language", None)
        self.date_of_snapshot = kwargs.pop("date", None)
        self.size = kwargs.pop("size", None)
        self.validated_hr_total = kwargs.pop("val_hrs", None)
        self.total_hr_total = kwargs.pop("total_hrs", None)
        self.num_of_voice = kwargs.pop("num_of_voice", None)
        
        self.unk_token_regex = None
        if self.language in hg.Languages.get_all():
            # creating regex to match language specific non valid characters
            currency_symbols = ["$", "£", "€", "¥", "₩", "₹", "₽", "₱", "₦", "₼", "ლ", "₭", "₴", "₲", "₫", "₡", "₵", "₿", "฿", "¢"]
            alphabet = list(hg.Languages.get_alphabet([self.language]))
            valid_chars = alphabet + currency_symbols
            self.unk_token_regex = "[^"+re.escape("".join(valid_chars))+"\s\d]"
        
        description = f"Common Voice speech to text dataset in {self.language} version {self.sub_version} of {self.date_of_snapshot}. The dataset comprises {self.validated_hr_total} of validated transcribed speech data from {self.num_of_voice} speakers. The dataset has a size of {self.size}"
        super(CommonVoiceConfig, self).__init__(
            name=name, version=datasets.Version("6.1.0", ""), description=description, **kwargs
        )


class CommonVoice(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        CommonVoiceConfig(
            name=lang_id,
            language=_LANGUAGES[lang_id]["Language"],
            sub_version=_LANGUAGES[lang_id]["Version"],
            date=_LANGUAGES[lang_id]["Date"],
            size=_LANGUAGES[lang_id]["Size"],
            val_hrs=_LANGUAGES[lang_id]["Validated_Hr_Total"],
            total_hrs=_LANGUAGES[lang_id]["Overall_Hr_Total"],
            num_of_voice=_LANGUAGES[lang_id]["Number_Of_Voice"],
        )
        for lang_id in _LANGUAGES.keys()
    ]

    def _info(self):
        features = datasets.Features(
            {
                "client_id": datasets.Value("string"),
                "path": datasets.Value("string"),
                "sentence": datasets.Value("string"),
                "up_votes": datasets.Value("int64"),
                "down_votes": datasets.Value("int64"),
                "age": datasets.Value("string"),
                "gender": datasets.Value("string"),
                "accent": datasets.Value("string"),
                "locale": datasets.Value("string"),
                "segment": datasets.Value("string"),
                "duration": datasets.Value("float32"),
                "dataset": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )
    
    def _download_from_gdrive(self, src_url: str, dst_path: str):
        """Downloading from Gdrive"""
        gdown.download(src_url, dst_path, quiet=False)

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        dl_path = dl_manager.download_and_extract(_DATA_URL.format(self.config.name))
        abs_path_to_data = os.path.join(dl_path, "cv-corpus-6.1-2020-12-11", self.config.name)
        abs_path_to_clips = os.path.join(abs_path_to_data, "clips")

        css10_dir = None
        if self.config.name in _CSS10_URLS:
            css10_url = _CSS10_URLS[self.config.name]
            css10_dir = dl_manager.extract(dl_manager.download_custom(css10_url, self._download_from_gdrive))

        jsut_dir = None
        if self.config.name in _JSUT_URLS:
            jsut_url = _JSUT_URLS[self.config.name]
            jsut_dir = dl_manager.download_and_extract(jsut_url)
            jsut_dir = os.path.join(jsut_dir, "jsut_ver1.1")

        nst_metadata_dir = None
        nst_files_dir = None
        if self.config.name in _NST_URLS:
            nst_metadata_dir = dl_manager.download_and_extract(_NST_URLS[self.config.name]["metadata"])
            nst_files_dir = dl_manager.download_and_extract(_NST_URLS[self.config.name]["files"])

        free_st_dir = None
        if self.config.name in _FREE_ST_URLS:
            free_st_dir = dl_manager.download_and_extract(_FREE_ST_URLS[self.config.name])
            free_st_dir = os.path.join(free_st_dir, "ST-CMDS-20170001_1-OS")

        arabic_speech_dir = None
        if self.config.name in _ARABIC_SPEECH:
            arabic_speech_dir = dl_manager.download_and_extract(_ARABIC_SPEECH[self.config.name])
            arabic_speech_dir = os.path.join(arabic_speech_dir, "arabic-speech-corpus")

        timit_dir = None
        if self.config.name in _TIMIT:
            timit_dir = dl_manager.download_and_extract(_TIMIT[self.config.name])

        librispeech_dirs = None
        if self.config.name in _LIBRISPEECH:
            librispeech_dirs = []
            for librispeech_url in _LIBRISPEECH[self.config.name]:
                librispeech_dir = dl_manager.download_and_extract(librispeech_url)
                librispeech_dirs.append(librispeech_dir)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(abs_path_to_data, "train.tsv"),
                    "path_to_clips": abs_path_to_clips,
                    "css10_dir": css10_dir,
                    "jsut_dir": jsut_dir,
                    "nst_metadata_dir": nst_metadata_dir,
                    "nst_files_dir": nst_files_dir,
                    "free_st_dir": free_st_dir,
                    "arabic_speech_dir": arabic_speech_dir,
                    "timit_dir": timit_dir,
                    "librispeech_dirs": librispeech_dirs,
                    "max_samples": _MAX_TRAIN_SAMPLES
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(abs_path_to_data, "dev.tsv"),
                    "path_to_clips": abs_path_to_clips,
                    "css10_dir": None,
                    "jsut_dir": None,
                    "nst_metadata_dir": None,
                    "nst_files_dir": None,
                    "free_st_dir": None,
                    "arabic_speech_dir": None,
                    "timit_dir": None,
                    "librispeech_dirs": None,
                    "max_samples": _MAX_VAL_SAMPLES
                },
            )
        ]

    def _convert_to_flac_and_save_it(self, path, delete_original_file=True):
        """We'll convert all the audio files to FLAC format to speedup the loading"""
        
        sample_path, sample_extension = os.path.splitext(path)
        new_path = f"{sample_path}.flac"

        if not os.path.isfile(new_path):
        
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                speech_array, sample_rate = librosa.load(path, sr=16_000)
            
            sf.write(new_path, speech_array, sample_rate)

            if delete_original_file:
                os.remove(path)

        return new_path

    def _common_voice_examples_generator(self, filepath, path_to_clips):

        data_fields = list(self._info().features.keys())
        path_idx = data_fields.index("path")

        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()

            for line in lines[1:]:
                field_values = line.strip().split("\t")

                # set absolute path for mp3 audio file
                field_values[path_idx] = os.path.join(path_to_clips, field_values[path_idx])

                # if data is incomplete, fill with empty values
                if len(field_values) < len(data_fields):
                    field_values += (len(data_fields) - len(field_values)) * ["''"]

                sample = {key: value for key, value in zip(data_fields, field_values)}

                new_path = self._convert_to_flac_and_save_it(sample.get("path"))
                speech_array, sampling_rate = sf.read(new_path)
                sample["duration"] = len(speech_array) / sampling_rate
                sample["path"] = new_path
                sample["dataset"] = "common_voice"

                if self.config.unk_token_regex is not None:
                    sample["sentence"] = re.sub(self.config.unk_token_regex, "<unk>", sample["sentence"])

                yield sample

    def _css10_examples_generator(self, css10_dir):

        with open(os.path.join(css10_dir, "transcript.txt"), encoding="utf-8") as f:
            lines = f.readlines()

            for line in lines:
                values = line.strip().split("|")

                audio_path = self._convert_to_flac_and_save_it(os.path.join(css10_dir, values[0]))
                text = values[1] if self.config.name in ["ja", "zh"] else values[2]
                text = re.sub("\s+", " ", text) # remove multiple spaces
                duration = float(values[3])

                if self.config.unk_token_regex is not None:
                    text = re.sub(self.config.unk_token_regex, "<unk>", text)

                yield {
                    "client_id": None,
                    "path": audio_path,
                    "sentence": text,
                    "up_votes": 0,
                    "down_votes": 0,
                    "age": None,
                    "gender": None,
                    "accent": None,
                    "locale": None,
                    "segment": None,
                    "duration": duration,
                    "dataset": "css10"
                }

    def _jsut_examples_generator(self, jsut_dir):

        for subset in os.listdir(jsut_dir):
            
            if not os.path.isdir(os.path.join(jsut_dir, subset)):
                continue
                
            transcript_path = os.path.join(jsut_dir, subset, "transcript_utf8.txt")

            with open(transcript_path, encoding="utf-8") as f:

                lines = f.readlines()

                for line in lines:

                    values = line.split(":")
                    audio_path = os.path.join(jsut_dir, subset, "wav", f"{values[0]}.wav")
                    text = values[1]
                    text = re.sub("\s+", " ", text) # remove multiple spaces

                    if self.config.unk_token_regex is not None:
                        text = re.sub(self.config.unk_token_regex, "<unk>", text)

                    new_audio_path = self._convert_to_flac_and_save_it(audio_path)
                    speech_array, sampling_rate = sf.read(new_audio_path)
                    duration = len(speech_array) / sampling_rate

                    yield {
                        "client_id": None,
                        "path": new_audio_path,
                        "sentence": text,
                        "up_votes": 0,
                        "down_votes": 0,
                        "age": None,
                        "gender": None,
                        "accent": None,
                        "locale": None,
                        "segment": None,
                        "duration": duration,
                        "dataset": "jsut"
                    }

    def _nst_examples_generator(self, nst_metadata_dir, nst_files_dir):

        for metadata_filename in os.listdir(nst_metadata_dir):
            
            metadata_filepath = os.path.join(nst_metadata_dir, metadata_filename)

            with open(metadata_filepath) as metadata_file:
                metadata = json.load(metadata_file)

                client_id = metadata.get("info", {}).get("Speaker_ID", None)
                age = metadata.get("info", {}).get("Age", None)
                gender = metadata.get("info", {}).get("Sex", None)
                lang = metadata.get("metadata").get("lang")
                pid = metadata.get("pid")
                audio_dir = os.path.join(nst_files_dir, lang, pid)

                for val_recording in metadata.get("val_recordings", []):

                    audio_filename = f"{pid}_{val_recording.get('file').replace('.wav', '-1.wav')}"
                    audio_path = os.path.join(audio_dir, audio_filename)
                    
                    # there are some missing files on the original dataset, so we need to handle this
                    if not os.path.isfile(audio_path): 
                        continue
                    
                    text = val_recording.get("text")
                    text = re.sub("\s+", " ", text) # remove multiple spaces

                    if self.config.unk_token_regex is not None:
                        text = re.sub(self.config.unk_token_regex, "<unk>", text)

                    new_audio_path = self._convert_to_flac_and_save_it(audio_path)
                    speech_array, sampling_rate = sf.read(new_audio_path)
                    duration = len(speech_array) / sampling_rate

                    yield {
                        "client_id": client_id,
                        "path": new_audio_path,
                        "sentence": text,
                        "up_votes": 0,
                        "down_votes": 0,
                        "age": age,
                        "gender": gender,
                        "accent": None,
                        "locale": None,
                        "segment": None,
                        "duration": duration,
                        "dataset": "nst"
                    }

    def _free_st_examples_generator(self, free_st_dir):

        for filename in os.listdir(free_st_dir):

            if filename.endswith(".wav"):

                audio_path = os.path.join(free_st_dir, filename)
                text_path = os.path.join(free_st_dir, filename.replace(".wav", ".txt"))

                with open(text_path, "r") as text_file:
                    text = text_file.read().replace("\n", "").strip()
                    text = re.sub("\s+", " ", text) # remove multiple spaces

                if self.config.unk_token_regex is not None:
                    text = re.sub(self.config.unk_token_regex, "<unk>", text)

                new_audio_path = self._convert_to_flac_and_save_it(audio_path)
                speech_array, sampling_rate = sf.read(new_audio_path)
                duration = len(speech_array) / sampling_rate

                yield {
                    "client_id": None,
                    "path": new_audio_path,
                    "sentence": text,
                    "up_votes": 0,
                    "down_votes": 0,
                    "age": None,
                    "gender": None,
                    "accent": None,
                    "locale": None,
                    "segment": None,
                    "duration": duration,
                    "dataset": "free_st"
                }

    def _arabic_speech_examples_generator(self, arabic_speech_dir):

        with open(os.path.join(arabic_speech_dir, "orthographic-transcript.txt"), encoding="utf-8") as f:
            
            lines = f.readlines()

            for line in lines:

                values = line.split('" "')
                filename = values[0].strip()[1:]
                text = values[1].strip()[:-1]
                audio_path = os.path.join(arabic_speech_dir, "wav", filename)

                # converting buckwalter format to arabic letters
                text = buckwalter.untransliterate(text)
                text = re.sub("\s+", " ", text) # remove multiple spaces

                if self.config.unk_token_regex is not None:
                    text = re.sub(self.config.unk_token_regex, "<unk>", text)

                new_audio_path = self._convert_to_flac_and_save_it(audio_path)
                speech_array, sampling_rate = sf.read(new_audio_path)
                duration = len(speech_array) / sampling_rate

                yield {
                    "client_id": None,
                    "path": new_audio_path,
                    "sentence": text,
                    "up_votes": 0,
                    "down_votes": 0,
                    "age": None,
                    "gender": None,
                    "accent": None,
                    "locale": None,
                    "segment": None,
                    "duration": duration,
                    "dataset": "arabic_speech"
                }

    def _timit_examples_generator(self, timit_dir):

        data_info_csv = os.path.join(timit_dir, "train_data.csv")

        """Generate examples from TIMIT archive_path based on the test/train csv information."""
        # Extract the archive path
        data_path = os.path.join(os.path.dirname(data_info_csv).strip(), "data")

        # Read the data info to extract rows mentioning about non-converted audio only
        data_info = pd.read_csv(data_info_csv, encoding="utf8")
        # making sure that the columns having no information about the file paths are removed
        data_info.dropna(subset=["path_from_data_dir"], inplace=True)

        # filter out only the required information for data preparation
        data_info = data_info.loc[(data_info["is_audio"]) & (~data_info["is_converted_audio"])]

        # Iterating the contents of the data to extract the relevant information
        for audio_idx in range(data_info.shape[0]):
            audio_data = data_info.iloc[audio_idx]

            # extract the path to audio
            wav_path = os.path.join(data_path, *(audio_data["path_from_data_dir"].split("/")))

            # extract transcript
            with open(wav_path.replace(".WAV", ".TXT"), "r", encoding="utf-8") as op:
                transcript = " ".join(op.readlines()[0].split()[2:])  # first two items are sample number

            new_audio_path = self._convert_to_flac_and_save_it(wav_path)
            speech_array, sampling_rate = sf.read(new_audio_path)
            duration = len(speech_array) / sampling_rate

            yield {
                "client_id": str(audio_data["speaker_id"]),
                "path": new_audio_path,
                "sentence": transcript,
                "up_votes": 0,
                "down_votes": 0,
                "age": None,
                "gender": None,
                "accent": audio_data["dialect_region"],
                "locale": None,
                "segment": None,
                "duration": duration,
                "dataset": "timit"
            }

    def _librispeech_examples_generator(self, librispeech_dir):

        transcripts_glob = os.path.join(librispeech_dir, "LibriSpeech", "*/*/*/*.txt")
        for transcript_file in sorted(glob.glob(transcripts_glob)):
            path = os.path.dirname(transcript_file)
            # with open(os.path.join(path, transcript_file), "r", encoding="utf-8") as f:
            with open(transcript_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    key, transcript = line.split(" ", 1)
                    audio_file = f"{key}.flac"
                    audio_file = os.path.join(path, audio_file)
                    speaker_id, chapter_id = [int(el) for el in key.split("-")[:2]]

                    speech_array, sampling_rate = sf.read(audio_file)
                    duration = len(speech_array) / sampling_rate

                    yield {
                        "client_id": str(speaker_id),
                        "path": audio_file,
                        "sentence": transcript,
                        "up_votes": 0,
                        "down_votes": 0,
                        "age": None,
                        "gender": None,
                        "accent": None,
                        "locale": None,
                        "segment": None,
                        "duration": duration,
                        "dataset": "librispeech"
                    }

    def _generate_examples(self, filepath, path_to_clips, css10_dir, jsut_dir, nst_metadata_dir, 
                           nst_files_dir, free_st_dir, arabic_speech_dir, timit_dir, 
                           librispeech_dirs, max_samples):
        """ Yields examples. """
        _id = 0

        for example in self._common_voice_examples_generator(filepath, path_to_clips):
            if _id == max_samples:
                break
            yield _id, example
            _id += 1

        if timit_dir is not None and _id < max_samples:
            for example in self._timit_examples_generator(timit_dir):
                if _id < max_samples:
                    yield _id, example
                    _id += 1
                else:
                    break

        if css10_dir is not None and _id < max_samples:
            for example in self._css10_examples_generator(css10_dir):
                if _id < max_samples:
                    yield _id, example
                    _id += 1
                else:
                    break

        if librispeech_dirs is not None and _id < max_samples:
            for librispeech_dir in librispeech_dirs:
                for example in self._librispeech_examples_generator(librispeech_dir):
                    if _id < max_samples:
                        yield _id, example
                        _id += 1
                    else:
                        break

        if jsut_dir is not None and _id < max_samples:
            for example in self._jsut_examples_generator(jsut_dir):
                if _id < max_samples:
                    yield _id, example
                    _id += 1
                else:
                    break

        if nst_files_dir is not None and _id < max_samples:
            for example in self._nst_examples_generator(nst_metadata_dir, nst_files_dir):
                if _id < max_samples:
                    yield _id, example
                    _id += 1
                else:
                    break

        if free_st_dir is not None and _id < max_samples:
            for example in self._free_st_examples_generator(free_st_dir):
                if _id < max_samples:
                    yield _id, example
                    _id += 1
                else:
                    break
        
        if arabic_speech_dir is not None and _id < max_samples:
            root_dirs = [arabic_speech_dir, os.path.join(arabic_speech_dir, "test set")]
            for root_dir in root_dirs:
                for example in self._arabic_speech_examples_generator(root_dir):
                    if _id < max_samples:
                        yield _id, example
                        _id += 1
                    else:
                        break
