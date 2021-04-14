# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors.
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
""" Character Error Ratio (CER) metric. """
import jiwer
import jiwer.transforms as tr
from typing import List
import datasets
import gc

_CITATION = """\
@inproceedings{inproceedings,
    author = {Morris, Andrew and Maier, Viktoria and Green, Phil},
    year = {2004},
    month = {01},
    pages = {},
    title = {From WER and RIL to MER and WIL: improved evaluation measures for connected speech recognition.}
}
"""
_DESCRIPTION = """\
Character error rate (CER) is a common metric of the performance of an automatic speech recognition system.
CER is similar to Word Error Rate (WER), but operate on character insted of word. Please refer to docs of WER for further information.
Character error rate can be computed as:
CER = (S + D + I) / N = (S + D + I) / (S + D + C)
where
S is the number of substitutions,
D is the number of deletions,
I is the number of insertions,
C is the number of correct characters,
N is the number of characters in the reference (N=S+D+C).
CER's output is always a number between 0 and 1. This value indicates the percentage of characters that were incorrectly predicted. The lower the value, the better the
performance of the ASR system with a CER of 0 being a perfect score.
"""
_KWARGS_DESCRIPTION = """
Computes CER score of transcribed segments against references.
Args:
    references: list of references for each speech input.
    predictions: list of transcribtions to score.
Returns:
    (float): the character error rate
Examples:
    >>> predictions = ["this is the prediction", "there is an other sample"]
    >>> references = ["this is the reference", "there is another one"]
    >>> cer = datasets.load_metric("cer")
    >>> cer_score = cer.compute(predictions=predictions, references=references)
    >>> print(cer_score)
    0.34146341463414637
"""
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class CER(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                }
            ),
            codebase_urls=["https://github.com/jitsi/jiwer/"],
            reference_urls=[
                "https://en.wikipedia.org/wiki/Word_error_rate",
            ],
        )
    def _compute(self, predictions, references, chunk_size=None):
        if chunk_size is None:
            preds = [char for seq in predictions for char in list(seq)]
            refs = [char for seq in references for char in list(seq)]
            return jiwer.wer(refs, preds)
        start = 0
        end = chunk_size
        H, S, D, I = 0, 0, 0, 0
        while start < len(references):
            preds = [char for seq in predictions[start:end] for char in list(seq)]
            refs = [char for seq in references[start:end] for char in list(seq)]
            chunk_metrics = jiwer.compute_measures(refs, preds)
            H = H + chunk_metrics["hits"]
            S = S + chunk_metrics["substitutions"]
            D = D + chunk_metrics["deletions"]
            I = I + chunk_metrics["insertions"]
            start += chunk_size
            end += chunk_size
            del preds
            del refs
            del chunk_metrics
            gc.collect()
        return float(S + D + I) / float(H + S + D)
