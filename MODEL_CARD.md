---
language: pt
datasets:
- common_voice
metrics:
- wer
- cer
tags:
- audio
- automatic-speech-recognition
- speech
- xlsr-fine-tuning-week
license: apache-2.0
model-index:
- name: XLSR Wav2Vec2 Portuguese by Jonatas Grosman
  results:
  - task: 
      name: Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Common Voice pt
      type: common_voice
      args: pt
    metrics:
      - name: Test WER
         type: wer
         value: {wer_result_on_test}
       - name: Test CER
         type: cer
         value: {cer_result_on_test} #TODO (IMPORTANT): replace {wer_result_on_test} with the WER error rate you achieved on the common_voice test set. It should be in the format XX.XX (don't add the % sign here). **Please** remember to fill out this value after you evaluated your model, so that your model appears on the leaderboard. If you fill out this model card before evaluating your model, please remember to edit the model card afterward to fill in your value
---

# Wav2Vec2-Large-XLSR-53-Portuguese

Fine-tuned [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) on Portuguese using the [Common Voice](https://huggingface.co/datasets/common_voice).
When using this model, make sure that your speech input is sampled at 16kHz.

The script used for training can be found here: https://github.com/jonatasgrosman/wav2vec2-sprint
## Usage

The model can be used directly (without a language model) as follows:

```python
import torch
import librosa
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

LANG_ID = "pt"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese"

test_dataset = load_dataset("common_voice", LANG_ID, split="test[:2%]")

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = librosa.load(batch["path"], sr=16_000)
    batch["speech"] = speech_array
    batch["sentence"] = batch["sentence"].upper()
    return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)
inputs = processor(test_dataset[:2]["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

predicted_ids = torch.argmax(logits, dim=-1)

print("Prediction:", processor.batch_decode(predicted_ids))
print("Reference:", test_dataset[:2]["sentence"])
```

## Evaluation

The model can be evaluated as follows on the Portuguese test data of Common Voice.

```python
import torch
import librosa
import re
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

LANG_ID = "pt"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese"
DEVICE = "cuda"

CHARS_TO_IGNORE = [",", "?", "¿", ".", "!", "¡", "-", ";", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞", 
                   "؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")", "[", "]",
                   "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。"]

test_dataset = load_dataset("common_voice", LANG_ID, split="test")
wer = load_metric("wer") # https://github.com/jonatasgrosman/wav2vec2-sprint/blob/main/wer.py
cer = load_metric("cer") # https://github.com/jonatasgrosman/wav2vec2-sprint/blob/main/cer.py

chars_to_ignore_regex = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
model.to(DEVICE)

# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_array_fn(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).upper()
    speech_array, sampling_rate = librosa.load(batch["path"], sr=16_000)
    batch["speech"] = speech_array
    return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)

# Preprocessing the datasets.
# We need to read the audio files as arrays
def evaluate(batch):
	inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

	with torch.no_grad():
		logits = model(inputs.input_values.to(DEVICE), attention_mask=inputs.attention_mask.to(DEVICE)).logits

	pred_ids = torch.argmax(logits, dim=-1)
	batch["pred_strings"] = processor.batch_decode(pred_ids)
	return batch

result = test_dataset.map(evaluate, batched=True, batch_size=32)

print("WER: {:2f}".format(100 * wer.compute(predictions=result["pred_strings"], references=result["sentence"], chunk_size=8000)))
print("CER: {:2f}".format(100 * cer.compute(predictions=result["pred_strings"], references=result["sentence"], chunk_size=8000)))
```

**Test Result**: 

- WER: XX.XX%

- CER: XX.XX%
