import os
import numpy as np
import datasets
import torch
import torchaudio
import soundfile as sf
import librosa

from audiomentations import (
    Compose,
    AddGaussianNoise,
    AddGaussianSNR,
    ClippingDistortion,
    FrequencyMask,
    Gain,
    LoudnessNormalization,
    Normalize,
    PitchShift,
    PolarityInversion,
    Shift,
    TimeMask,
    TimeStretch,
)

os.makedirs("_ignore_data", exist_ok=True)

# creating augmentation pipeline

augmentator = Compose([
    AddGaussianNoise(min_amplitude=0.0001, max_amplitude=0.001, p=0.5),
    Gain(min_gain_in_db=-1, max_gain_in_db=1, p=0.5),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    # TimeStretch(min_rate=0.7, max_rate=1.3, leave_length_unchanged=False, p=0.5),
    # FrequencyMask(min_frequency_band=0.0, max_frequency_band=0.5, p=0.5),
    # TimeMask(min_band_part=0.0, max_band_part=0.01, fade=True, p=0.5),
    # ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=5, p=0.5),
    # LoudnessNormalization(min_lufs_in_db=-31, max_lufs_in_db=-13, p=0.5)
    # PolarityInversion(p=0.5),
    # Shift(min_fraction=-0.01, max_fraction=0.01, rollover=False, p=0.5),
    # AddGaussianSNR(min_SNR=0.001, max_SNR=1.0, p=0.5),
    # Normalize(p=0.5),
])

# Get the dataset

# train_dataset = datasets.load_dataset(
#     "common_voice_ext.py", "fi", augmentation_factor=2, split="train+validation", cache_dir="_ignore_data/cache/fi"
# )

train_dataset = datasets.load_dataset(
    "common_voice", "pt", split="train+validation", cache_dir="_ignore_data/cache/pt"
)

# Getting one sample per gender

male_sample = None
female_sample = None

for sample in train_dataset: 

    if sample.get("gender") == "male":
        male_sample = sample
    elif sample.get("gender") == "female":
        female_sample = sample
    
    if male_sample is not None and female_sample is not None:
        break

# Augmenting data and saving it

speech_array, sample_rate = librosa.load(male_sample["path"], sr = 16000, res_type="zero_order_hold")
speech_array_augmented = augmentator(samples=speech_array, sample_rate=sample_rate)
sf.write("_ignore_data/male_original.wav", speech_array, sample_rate, subtype="PCM_24")
sf.write("_ignore_data/male_augmented.flac", speech_array_augmented, sample_rate)

speech_array, sample_rate = librosa.load(female_sample["path"], sr = 16000, res_type="zero_order_hold")
speech_array_augmented = augmentator(samples=speech_array, sample_rate=sample_rate)
sf.write("_ignore_data/female_original.wav", speech_array, sample_rate, subtype="PCM_24")
sf.write("_ignore_data/female_augmented.wav", speech_array_augmented, sample_rate, subtype="PCM_24")

print(":)")
