#!/usr/bin/env python3
import json
import logging
import os
import re
import sys
import collections
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import datasets
import numpy as np
import torch
import soundfile as sf
from packaging import version
from torch import nn
from pathlib import Path
import wandb

from torch_audiomentations import Compose, Gain
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

import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    is_apex_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.trainer_pt_utils import LengthGroupedSampler, DistributedLengthGroupedSampler


PRETRAINED_MODELS = [
    "facebook/wav2vec2-large",
    "facebook/wav2vec2-large-xlsr-53",
    "facebook/wav2vec2-large-es-voxpopuli",
    "facebook/wav2vec2-large-fr-voxpopuli",
    "facebook/wav2vec2-large-it-voxpopuli",
    "facebook/wav2vec2-large-nl-voxpopuli",
    "facebook/wav2vec2-large-sv-voxpopuli",
    "facebook/wav2vec2-large-10k-voxpopuli",
    "facebook/wav2vec2-large-100k-voxpopuli"
]


if is_apex_available():
    from apex import amp


if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


@dataclass
class AdditionalTrainingArguments:
    """
    Additional training arguments
    """

    lr_warmup_ratio: Optional[float] = field(
        default=0.1,
        metadata={"help": "Percentage of steps for LR warmup phase"},
    )
    lr_constant_ratio: Optional[float] = field(
        default=0.4,
        metadata={"help": "Percentage of steps for LR constant phase (after warmup)"},
    )
    upload_final_model_to_wandb: Optional[bool] = field(
        default=False,
        metadata={"help": "Upload the final trained model to the WandB artifacts repository"},
    )
    upload_model_to_wandb_each_step: Optional[int] = field(
        default=None,
        metadata={"help": "Frequency (in steps) to upload the trained model to the WandB artifacts repository"},
    )
    apply_gaussian_noise_with_p: Optional[float] = field(
        default=0.5,
        metadata={"help": "Probability to apply Gaussian Noise in the original samples"},
    )
    apply_gain_with_p: Optional[float] = field(
        default=0.5,
        metadata={"help": "Probability to apply Gain in the original samples"},
    )
    apply_pitch_shift_with_p: Optional[float] = field(
        default=0.5,
        metadata={"help": "Probability to apply Pitch Shift in the original samples"},
    )
    apply_time_stretch_with_p: Optional[float] = field(
        default=0.5,
        metadata={"help": "Probability to apply Time Stretch in the original samples"},
    )
    min_char_occurrence_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "Minimum ratio of character occurrences to be considered for the vocabulary builder"},
    )
    max_dataset_size_vocab_builder: Optional[int] = field(
        default=10000,
        metadata={"help": "Maximum size of the dataset to be considered for vocabulary builder"},
    )
    remove_samples_with_oov_from_training: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to remove samples from training when there are OOV characters on them"},
    )
    use_only_top_k_most_common_accent: Optional[int] = field(
        default=None,
        metadata={"help": "Use only the top most common accent in dataset for training"},
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=True, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    attention_dropout: Optional[float] = field(
        default=0.1, metadata={"help": "The dropout ratio for the attention probabilities."}
    )
    activation_dropout: Optional[float] = field(
        default=0.1, metadata={"help": "The dropout ratio for activations inside the fully connected layer."}
    )
    hidden_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler."
        },
    )
    feat_proj_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "The dropout probabilitiy for all 1D convolutional layers in feature extractor."},
    )
    mask_time_prob: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "Propability of each feature vector along the time axis to be chosen as the start of the vector"
            "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature"
            "vectors will be masked along the time axis. This is only relevant if ``apply_spec_augment is True``."
        },
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    layerdrop: Optional[float] = field(default=0.0, metadata={"help": "The LayerDrop probability."})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=1000,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    val_ratio: Optional[float] = field(
        default=0.2,
        metadata={
            "help": "Percentage of dataset samples to be used for evaluation, default is 20%"
        },
    )
    chars_to_ignore: List[str] = list_field(
        default=[",", "?", "¿", ".", "!", "¡", ";", "；", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞",
                 "؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")", "[", "]",
                 "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。",
                 "、", "﹂", "﹁", "‧", "～", "﹏", "，", "｛", "｝", "（", "）", "［", "］", "【", "】", "‥", "〽",
                 "『", "』", "〝", "〟", "⟨", "⟩", "〜", "：", "！", "？", "♪", "؛", "/", "\\", "º", "−", "^", "ʻ", "ˆ"],
        metadata={"help": "A list of characters to remove from the transcripts."},
    )
    min_duration: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "The minimum duration (in seconds) that a sample needs to have to be considered for training"
        },
    )
    max_duration: Optional[float] = field(
        default=float("inf"),
        metadata={
            "help": "The maximum duration (in seconds) that a sample needs to have to be considered for training"
        },
    )
    use_only_common_voice_data: bool = field(
        default=False, metadata={"help": "Use only common voice data in training."}
    )


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __init__(self, processor, padding=True, apply_gaussian_noise_with_p=0.5, apply_gain_with_p=0.5, apply_pitch_shift_with_p=0.5, 
                 apply_time_stretch_with_p=0.5, sample_rate=16_000):
        self.processor = processor
        self.padding = padding
        self.apply_gaussian_noise_with_p = apply_gaussian_noise_with_p
        self.apply_gain_with_p = apply_gain_with_p
        self.apply_pitch_shift_with_p = apply_pitch_shift_with_p
        self.apply_time_stretch_with_p = apply_time_stretch_with_p
        self.sample_rate = sample_rate

        self.augmentator = None
        if self.apply_gaussian_noise_with_p + self.apply_gain_with_p + self.apply_pitch_shift_with_p + self.apply_time_stretch_with_p > 0:
            self.augmentator = Compose([
                TimeStretch(min_rate=0.8, max_rate=1.2, leave_length_unchanged=False, p=self.apply_time_stretch_with_p),
                PitchShift(min_semitones=-1, max_semitones=1, p=self.apply_pitch_shift_with_p),
                Gain(min_gain_in_db=-1, max_gain_in_db=1, p=self.apply_gain_with_p),
                AddGaussianNoise(min_amplitude=0.0001, max_amplitude=0.001, p=self.apply_gaussian_noise_with_p),
            ])

    def _apply_augmentation(self, input_values: List[float]):
        """apply some audio augmentations in the given input_values"""
        if self.augmentator is not None:
            return self.augmentator(samples=np.array(input_values), sample_rate=self.sample_rate).tolist()
        else:
            return input_values

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods

        input_features = [{"input_values": self._apply_augmentation(feature["input_values"])} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


class CTCTrainer(Trainer):

    def __init__(self, model_output_dir, length_field_name="length", upload_model_to_wandb_each_step=None, lr_warmup_ratio=0.1, 
                lr_constant_ratio=0.4, sampling_rate=16_000, **kwargs):
        super().__init__(**kwargs)
        self.model_output_dir = model_output_dir
        self.length_field_name = length_field_name
        self.upload_model_to_wandb_each_step = upload_model_to_wandb_each_step
        self.lr_warmup_ratio = lr_warmup_ratio
        self.lr_constant_ratio = lr_constant_ratio
        self.sampling_rate = sampling_rate

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
            self.train_dataset, collections.abc.Sized
        ):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            lengths = self.train_dataset[self.length_field_name] if self.length_field_name is not None else None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.train_dataset, self.args.train_batch_size, lengths=lengths, model_input_name=model_input_name
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.train_dataset,
                    self.args.train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                )

        else:
            return super()._get_train_sampler()

    def create_scheduler(self, num_training_steps: int):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up before this method is called.

        This method was built based on https://arxiv.org/pdf/2006.13979 :
            "The learning rate schedule has three phases: warm up for the first 10% of updates, 
             keep constant for 40% and then linearly decay for the remainder"
        
        Args:
            num_training_steps (int): The number of training steps to do.
        """
        def lr_lambda(current_step):
            warmup_steps = int(num_training_steps * self.lr_warmup_ratio)
            constant_steps = int(num_training_steps * self.lr_constant_ratio)
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            elif (self.lr_warmup_ratio + self.lr_constant_ratio) == 1.0 or current_step < (warmup_steps + constant_steps):
                return 1
            else: 
                return max(
                    0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - (warmup_steps + constant_steps)))
                )
        
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _apply_some_audio_transformations(self, inputs):
        """Perform some audio transformations"""
        
        # adding an extra dimmention for the channels as our data is mono audio and
        # the expected shape of input for torch_audiomentations is (batch_size, num_channels, num_samples)
        transformed_inputs = inputs["input_values"].unsqueeze(1)

        transformed_inputs = self.augmentator(transformed_inputs, sample_rate=self.sampling_rate)
           
        # returning the inputs to the original shape
        transformed_inputs = torch.squeeze(transformed_inputs, 1)
        
        inputs["input_values"] = transformed_inputs

        return inputs

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            if model.module.config.ctc_loss_reduction == "mean":
                loss = loss.mean()
            elif model.module.config.ctc_loss_reduction == "sum":
                loss = loss.sum() / (inputs["labels"] >= 0).sum()
            else:
                raise ValueError(f"{model.config.ctc_loss_reduction} is not valid. Choose one of ['mean', 'sum']")

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        if self.upload_model_to_wandb_each_step is not None and self.state.global_step > 0 \
            and self.state.global_step % self.upload_model_to_wandb_each_step == 0:
            upload_model_to_wandb(self.model_output_dir, name=f"{wandb.run.name}_{self.state.global_step}", metadata={"loss": float(loss)})

        return loss.detach()


def build_tokenizer(model_output_dir, dataset, num_proc, min_char_occurrence_ratio):

    def extract_all_chars(batch):
        all_text = " ".join(batch["text"]).replace("<unk>", "")
        return {"all_text": [all_text]}

    vocab_train = dataset.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        remove_columns=dataset.column_names,
        num_proc=num_proc
    )

    special_vocab_dict = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4}

    min_char_occurrence = int(min_char_occurrence_ratio * len(vocab_train["all_text"][0])) if min_char_occurrence_ratio is not None else 1

    if min_char_occurrence > 1:
        character_counter = collections.Counter(vocab_train["all_text"][0])
        vocab_list = [character for character, count in character_counter.items() if count >= min_char_occurrence]
    else:
        vocab_list = set(vocab_train["all_text"][0])

    vocab_list = [x for x in vocab_list if x.isalpha() or x in ["-", "'"]] # removing non-alpha (except - or ') characters

    vocab_list = sorted(vocab_list)
    vocab_dict = {v: k + len(special_vocab_dict) for k, v in enumerate(vocab_list)}
    vocab_dict = dict(special_vocab_dict, **vocab_dict)

    vocab_path = os.path.join(model_output_dir, "vocab.json")

    with open(vocab_path, "w") as vocab_file:
        json.dump(vocab_dict, vocab_file)

    return Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
    )


def upload_model_to_wandb(model_output_dir, name, metadata=None):
    artifact = wandb.Artifact(name=name, type="model", metadata=metadata)
    artifact.add_dir(model_output_dir)
    wandb.run.log_artifact(artifact)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # override default run name

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, AdditionalTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, additional_training_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, additional_training_args, training_args = parser.parse_args_into_dataclasses()

    os.makedirs(training_args.output_dir, exist_ok=True)
    os.makedirs(model_args.cache_dir, exist_ok=True)
    
    wandb.init(dir=model_args.cache_dir)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets:

    # As Common Voice dataset for most of the languages are really small, we'll merge the train and validation splits
    dataset = datasets.load_dataset(
        "dataset_ext.py", data_args.dataset_config_name,
        split="train+validation", 
        cache_dir=model_args.cache_dir
    )
    
    print("DATASET COUNT:")
    print(collections.Counter(dataset["dataset"]))

    if data_args.val_ratio > 0 and data_args.max_val_samples > 0 and training_args.do_eval:
        if len(dataset) * data_args.val_ratio > data_args.max_val_samples:
            dataset = dataset.train_test_split(test_size=data_args.max_val_samples)
        else:
            dataset = dataset.train_test_split(test_size=data_args.val_ratio)

        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

    else:
        train_dataset = dataset
        eval_dataset = None


    # Filtering dataset:
    
    train_dataset_original_size = len(train_dataset)
    if eval_dataset is not None:
        eval_dataset_original_size = len(eval_dataset)

    if data_args.use_only_common_voice_data:
        train_dataset = train_dataset.filter(
            lambda example: example["dataset"] == "common_voice",
            num_proc=data_args.preprocessing_num_workers
        )

    train_dataset = train_dataset.filter(
        lambda example: example["duration"] >= data_args.min_duration and example["duration"] <= data_args.max_duration,
        num_proc=data_args.preprocessing_num_workers
    )
    
    if data_args.max_train_samples is not None and train_dataset_original_size > data_args.max_train_samples:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if eval_dataset is not None and data_args.max_val_samples is not None and eval_dataset_original_size > data_args.max_val_samples:
        eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    train_dataset_final_size = len(train_dataset)
    if eval_dataset is not None:
        eval_dataset_final_size = len(eval_dataset)
    
    logger.info(f"After filtering {train_dataset_final_size} of {train_dataset_original_size} samples will be used to train the model")
    if eval_dataset is not None:
        logger.info(f"After filtering {eval_dataset_final_size} of {eval_dataset_original_size} samples will be used to eval the model")

    # Create and save tokenizer
    chars_to_ignore_regex = f"[{re.escape(''.join(data_args.chars_to_ignore))}]"

    def remove_special_characters(batch):
        batch["text"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).strip().upper() + " "
        return batch

    train_dataset = train_dataset.map(
        remove_special_characters,
        remove_columns=["sentence"], 
        num_proc=data_args.preprocessing_num_workers
    )
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            remove_special_characters, 
            remove_columns=["sentence"], 
            num_proc=data_args.preprocessing_num_workers
        )
    
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    
    if model_args.model_name_or_path in PRETRAINED_MODELS:
        dataset = datasets.concatenate_datasets([train_dataset, eval_dataset]) if eval_dataset is not None else train_dataset
        if len(dataset) > additional_training_args.max_dataset_size_vocab_builder:
            dataset = dataset.select(range(additional_training_args.max_dataset_size_vocab_builder))
        tokenizer = build_tokenizer(training_args.output_dir, dataset, data_args.preprocessing_num_workers, additional_training_args.min_char_occurrence_ratio)
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1, sampling_rate=16_000, padding_value=0.0, do_normalize=True, return_attention_mask=True
        )
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    else:
        processor = Wav2Vec2Processor.from_pretrained(model_args.model_name_or_path)

    if additional_training_args.remove_samples_with_oov_from_training:
        vocab = set(processor.tokenizer.encoder.keys())
        train_dataset_size = len(train_dataset)
        train_dataset = train_dataset.filter(
            lambda example: vocab.issuperset(example["text"].replace(" ", "")),
            num_proc=data_args.preprocessing_num_workers
        )
        print(f"OOV found in {train_dataset_size - len(train_dataset)} samples, and they were removed from training set")
        print(f"The final training set size is {len(train_dataset)}")

    if additional_training_args.use_only_top_k_most_common_accent is not None:

        train_dataset_size = len(train_dataset)

        accent_count = collections.Counter(train_dataset["accent"])
        # accent_count.pop("", None)
        major_accents = [k for k, x in accent_count.most_common(additional_training_args.use_only_top_k_most_common_accent)]

        print(f"ACCENT COUNT: {accent_count}")

        train_dataset = train_dataset.filter(
            lambda example: example["accent"] in major_accents,
            num_proc=data_args.preprocessing_num_workers
        )

        print(f"{train_dataset_size - len(train_dataset)} were removed from dataset due accent filtering, the final training dataset size is {len(train_dataset)}")

    # save the feature_extractor and the tokenizer
    processor.save_pretrained(training_args.output_dir)
    
    model = Wav2Vec2ForCTC.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        activation_dropout=model_args.activation_dropout,
        attention_dropout=model_args.attention_dropout,
        hidden_dropout=model_args.hidden_dropout,
        feat_proj_dropout=model_args.feat_proj_dropout,
        mask_time_prob=model_args.mask_time_prob,
        gradient_checkpointing=model_args.gradient_checkpointing,
        layerdrop=model_args.layerdrop,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ctc_zero_infinity=True
    )

    # Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    def speech_file_to_array_fn(batch):
        speech_array, sampling_rate = sf.read(batch["path"])
        batch["speech"] = speech_array
        batch["sampling_rate"] = sampling_rate
        batch["target_text"] = batch["text"]
        return batch
    
    print("TRAIN DATASET COUNT:")
    print(collections.Counter(train_dataset["dataset"]))
    print("EVAL DATASET COUNT:")
    print(collections.Counter(eval_dataset["dataset"]))

    train_dataset = train_dataset.map(
        speech_file_to_array_fn,
        remove_columns=train_dataset.column_names,
        num_proc=data_args.preprocessing_num_workers
    )
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            speech_file_to_array_fn,
            remove_columns=eval_dataset.column_names,
            num_proc=data_args.preprocessing_num_workers
        )

    def prepare_dataset(batch):
        # check that all files have the correct sampling rate
        assert (
            len(set(batch["sampling_rate"])) == 1
        ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."
        batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
        
        # Setup the processor for targets
        with processor.as_target_processor():
            batch["labels"] = processor(batch["target_text"]).input_ids
        return batch

    train_dataset = train_dataset.map(
        prepare_dataset,
        remove_columns=train_dataset.column_names,
        batch_size=training_args.per_device_train_batch_size,
        batched=True,
        num_proc=data_args.preprocessing_num_workers
    )
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            prepare_dataset,
            remove_columns=eval_dataset.column_names,
            batch_size=training_args.per_device_train_batch_size,
            batched=True,
            num_proc=data_args.preprocessing_num_workers
        )

    # Pre-compute sample lengths
    def input_lengths(example):
        example["length"] = len(example["input_values"])
        return example

    train_dataset = train_dataset.map(
        input_lengths,
        num_proc=data_args.preprocessing_num_workers
    )

    # Metric
    wer_metric = datasets.load_metric("wer.py")
    cer_metric = datasets.load_metric("cer.py")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str, chunk_size=1000)
        cer = cer_metric.compute(predictions=pred_str, references=label_str, chunk_size=1000)

        return {"wer": wer, "cer": cer}

    if model_args.freeze_feature_extractor:
        model.freeze_feature_extractor()

    # Data collator
    data_collator = DataCollatorCTCWithPadding(
        processor=processor,
        padding=True,
        apply_gaussian_noise_with_p=additional_training_args.apply_gaussian_noise_with_p, 
        apply_gain_with_p=additional_training_args.apply_gain_with_p, 
        apply_pitch_shift_with_p=additional_training_args.apply_pitch_shift_with_p,
        apply_time_stretch_with_p=additional_training_args.apply_time_stretch_with_p, 
        sample_rate=16_000,
    )

    # Initialize our Trainer
    trainer = CTCTrainer(
        model_output_dir=training_args.output_dir,
        length_field_name="length",
        upload_model_to_wandb_each_step=additional_training_args.upload_model_to_wandb_each_step,
        lr_warmup_ratio=additional_training_args.lr_warmup_ratio, 
        lr_constant_ratio=additional_training_args.lr_constant_ratio,
        sampling_rate=16_000,
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    metrics = {}
    if eval_dataset is not None and training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # save model files
    if additional_training_args.upload_final_model_to_wandb:
        upload_model_to_wandb(training_args.output_dir, name=f"{wandb.run.name}_final", metadata=metrics)

if __name__ == "__main__":
    main()
