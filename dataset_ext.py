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

import datasets

import soundfile as sf
import librosa
import warnings

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
    "de": "https://drive.google.com/uc?id=1wgCHGvT0S8YrNfRTVyn23sW-5MFknoHA",
    "el": "https://drive.google.com/uc?id=10BNORyOqkosxEf3qAAtWM1qWjHEZzXTO",
    "es": "https://drive.google.com/uc?id=1dyUvSxv0KowTseI35dE8UXpVsYFhEpQV",
    "fi": "https://drive.google.com/uc?id=1H4-eGIgf4aK_s14uo-srbKMENpysuV2u",
    "fr": "https://drive.google.com/uc?id=1kuhoDjhA_Cij0SJuMI_4kneDTR_cqahS",
    "hu": "https://drive.google.com/uc?id=1ms2INJ1e0ChU0TMzgDYLa8jtoTK2gkmE",
    "ja": "https://drive.google.com/uc?id=1E4k8FduAk-_wy85AQrGakZBcw2hLhmU6",
    "nl": "https://drive.google.com/uc?id=1ji8QD4lJzInz2vomGkMafRjpz3gGBYsf",
    "ru": "https://drive.google.com/uc?id=1tx3dpO8SX8CriF0YsK8XeISZc9yGRody",
    "zh-CN": "https://drive.google.com/uc?id=1hliY4KD_I8y4FQg5zta9IDGN0HRQLRiv",
}

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

        css10_dl_path = None
        if self.config.name in _CSS10_URLS:
            css10_url = _CSS10_URLS[self.config.name]
            css10_dl_path = dl_manager.extract(dl_manager.download_custom(css10_url, self._download_from_gdrive))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(abs_path_to_data, "train.tsv"),
                    "path_to_clips": abs_path_to_clips,
                    "css10_dir": css10_dl_path
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(abs_path_to_data, "test.tsv"),
                    "path_to_clips": abs_path_to_clips,
                    "css10_dir": None
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(abs_path_to_data, "dev.tsv"),
                    "path_to_clips": abs_path_to_clips,
                    "css10_dir": None
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

    def _generate_examples(self, filepath, path_to_clips, css10_dir):
        """ Yields examples. """
        _id = 0

        for example in self._common_voice_examples_generator(filepath, path_to_clips):
            yield _id, example
            _id += 1

        if css10_dir is not None:
            for example in self._css10_examples_generator(css10_dir):
                yield _id, example
                _id += 1
