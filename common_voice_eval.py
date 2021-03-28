import torch
import torchaudio
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import re
import homoglyphs as hg

LANG_ID = "fi"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-finnish"
DEVICE = "cuda"

CHARS_TO_IGNORE = [",", "?", ".", "!", "-", ";", ":", '""', "%", "'", '"', "�", "ʿ", "·", "჻", "¿", "¡", "~", "՞", 
                   "؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")", "[", "]",
                   "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", ""]
CURRENCY_SYMBOLS = ["$", "£", "€", "¥", "₩", "₹", "₽", "₱", "₦", "₼", "ლ", "₭", "₴", "₲", "₫", "₡", "₵", "₿", "฿", "¢"]

test_dataset = load_dataset("common_voice", LANG_ID, split="test")
wer = load_metric("wer")

unk_regex = None
if LANG_ID in hg.Languages.get_all():
    # creating regex to match language specific non valid characters
    alphabet = list(hg.Languages.get_alphabet([LANG_ID]))
    valid_chars = alphabet + CURRENCY_SYMBOLS
    unk_regex = "[^"+re.escape("".join(valid_chars))+"\s\d]"

chars_to_ignore_regex = f'[{re.escape("".join(CHARS_TO_IGNORE))}]'

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
model.to(DEVICE)

resampler = torchaudio.transforms.Resample(48_000, 16_000)

# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_array_fn(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).upper()
    if unk_regex is not None:
        batch["sentence"] = re.sub(unk_regex, "[UNK]", batch["sentence"])
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = resampler(speech_array).squeeze().numpy()
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

print("WER: {:2f}".format(100 * wer.compute(predictions=result["pred_strings"], references=result["sentence"])))