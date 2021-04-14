import torch
import librosa
import re
import warnings
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

LANG_ID = "pt"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese"
DEVICE = "cuda"
MAX_SAMPLES = 8000

CHARS_TO_IGNORE = [",", "?", "¿", ".", "!", "¡", ";", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞", 
                   "؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")", "[", "]",
                   "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。"]

test_dataset = load_dataset("common_voice", LANG_ID, split="test")
if len(test_dataset) > MAX_SAMPLES:
    test_dataset = test_dataset.select(range(MAX_SAMPLES))

wer = load_metric("wer.py") # https://github.com/jonatasgrosman/wav2vec2-sprint/blob/main/wer.py
cer = load_metric("cer.py") # https://github.com/jonatasgrosman/wav2vec2-sprint/blob/main/cer.py

chars_to_ignore_regex = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
model.to(DEVICE)

# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_array_fn(batch):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        speech_array, sampling_rate = librosa.load(batch["path"], sr=16_000)
    batch["speech"] = speech_array
    batch["sentence"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).upper()
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

result = test_dataset.map(evaluate, batched=True, batch_size=8)

print("WER: {:2f}".format(100 * wer.compute(predictions=result["pred_strings"], references=result["sentence"], chunk_size=1000)))
print("CER: {:2f}".format(100 * cer.compute(predictions=result["pred_strings"], references=result["sentence"], chunk_size=1000)))