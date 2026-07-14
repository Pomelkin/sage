"""Abstract API to spelling correction models.

The file also contains available pre-trained models for spelling correction
in Russian and English (yet more is to come).

To see all available models:

    models = [model.name for model in AvailableCorrectors]

To launch one of the available models:

    model_path = AvailableCorrectors.m2m100_1B.value
    ... # pass model path for initialization

"""

import os
import enum
import yaml
from accelerate import Accelerator
from abc import ABCMeta, abstractmethod
from typing import List, Union, Dict, Optional, Any

import pandas as pd
from torch.utils.data import DataLoader
from transformers import T5PreTrainedModel, T5ForConditionalGeneration

from .training.data_processor import get_tokenized_datasets, TextCollatorWithPadding
from .training.trainer import SageTrainer
from ..utils.data_load_utils import load_available_dataset_from_hf, DatasetsAvailable
from .models import T5ForConditionalGenerationTokenMultilabel, T5ForConditionalGenerationTokenMulticlass, \
    T5ForConditionalGenerationTokenMultilabelLM, T5ForConditionalGenerationLM

datasets_available = [dataset.name for dataset in DatasetsAvailable]
models_available = [T5PreTrainedModel]
models_names = {
    't5_encoder_multilabel': T5ForConditionalGenerationTokenMultilabel,
    't5_encoder_multiclass': T5ForConditionalGenerationTokenMulticlass,
    't5_encoder_multilabel_lm': T5ForConditionalGenerationTokenMultilabelLM,
    't5_encoder_lm': T5ForConditionalGenerationLM,
    't5': T5ForConditionalGeneration
}


class AvailableCorrectors(enum.Enum):
    """Available models for spelling and punctuation correction"""

    sage_fredt5_large = "ai-forever/sage-fredt5-large"
    sage_fredt5_distilled_95m = "ai-forever/sage-fredt5-distilled-95m"
    sage_m2m100_1B = "ai-forever/sage-m2m100-1.2B"
    sage_mt5_large = "ai-forever/sage-mt5-large"

    m2m100_1B = "ai-forever/RuM2M100-1.2B"
    m2m100_418M = "ai-forever/RuM2M100-418M"
    fred_large = "ai-forever/FRED-T5-large-spell"
    ent5_large = "ai-forever/T5-large-spell"


class Corrector(metaclass=ABCMeta):
    """Base class for all correctors."""

    @classmethod
    @abstractmethod
    def from_pretrained(cls, model_name_or_path: Union[str, os.PathLike]):
        pass

    def correct(self, sentence: str, prefix: Optional[str] = "", **generation_params) -> List[str]:
        """
        Corrects a single input sentence.

        :param sentence: a source sentence;
        :type sentence: str
        :param prefix: some models need some sort of a prompting;
        :type prefix: str
        :param generation_params: parameters passed to `generate` method of a HuggingFace model;
        :type generation_params: dict
        :return: corresponding corrected sentence
        :rtype: list of str
        """
        return self.batch_correct([sentence], 1, prefix, **generation_params)[-1]

    def evaluate(
            self,
            dataset_name_or_path: Optional[Union[str, os.PathLike]],
            metrics: List,
            batch_size: int,
            prefix: str = "",
            dataset_split: str = "test",
            **generation_params,
    ) -> Dict[str, float]:
        """
        Evaluate the particular model on the spellcheck datasets.

        :param dataset_name_or_path: a path to a locally situated dataset or a name of a dataset on HuggingFace;
        :type dataset_name_or_path: str
        :param metrics: set of metrics to be used to report performance;
        :type metrics: list of str
        :param batch_size: size of subsample of input sentences;
        :type batch_size: int
        :param prefix: some models need some sort of a prompting;
        :type prefix: str
        :param dataset_split: train / test / dev part to be evaluated on;
        :type dataset_split: str
        :param generation_params: parameters passed to `generate` method of a HuggingFace model;
        :type generation_params: dict
        :return: mapping between metric's name and its corresponding value
        :rtype: dict[str, float]
        """
        from ..evaluation.scorer import Scorer
        dataset_name_or_path = str(dataset_name_or_path)
        if dataset_name_or_path in datasets_available:
            sources, corrections = load_available_dataset_from_hf(
                dataset_name_or_path, for_labeler=True, split=dataset_split)
        elif os.path.isdir(dataset_name_or_path):
            if os.path.isfile(os.path.join(dataset_name_or_path, "sources.txt")) and \
                    os.path.isfile(os.path.join(dataset_name_or_path, "corrections.txt")):
                src_file = open(os.path.join(dataset_name_or_path, "sources.txt"), encoding="utf8")
                corr_file = open(os.path.join(dataset_name_or_path, "corrections.txt"), encoding="utf8")
                sources = src_file.read().split("\n")
                corrections = corr_file.read().split("\n")
                src_file.close()
                corr_file.close()
                if len(sources) != len(corrections):
                    raise RuntimeError("Sources and corrections must be of the same length, but get {} vs {}".format(
                        len(sources), len(corrections)))
            elif os.path.isfile(os.path.join(dataset_name_or_path, "data.csv")):
                try:
                    data = pd.read_csv(os.path.join(dataset_name_or_path, "data.csv"))
                except Exception as e:
                    raise RuntimeError("Wrong format of file {}. Raised an error: {}".format(
                        os.path.join(dataset_name_or_path, "data.csv"), str(e)))
                if not ("source" in data and "correction" in data):
                    raise RuntimeError("You must provide 'source' and 'correction' columns in {}".format(
                        os.path.join(dataset_name_or_path, "data.csv")
                    ))
                if data.isna().any().max():
                    raise ValueError("Your data at {} contain unnecessary nans".format(
                        os.path.join(dataset_name_or_path, "data.csv")))
                sources = data.source.values.tolist()
                corrections = data.correction.values.tolist()
            else:
                raise RuntimeError("You must provide either 'data.csv' or 'sources.txt'/'corrections.txt' in {}".format(
                    dataset_name_or_path
                ))
        else:
            raise ValueError("You must provide either valid path or available dataset's name, you provided {}".format(
                dataset_name_or_path
            ))

        answers = self.batch_correct(sources, batch_size, prefix, **generation_params)
        if "num_return_sequences" in generation_params and generation_params["num_return_sequences"] > 1:
            num_sequences = generation_params["num_return_sequences"]
            answers = [batch_answers[::num_sequences] for batch_answers in answers]
        answers = sum(answers, [])
        scorer = Scorer("errant" in metrics)
        metrics_dict = scorer.score(sources, corrections, answers, metrics)
        return metrics_dict

    @abstractmethod
    def batch_correct(
            self,
            sentences: List[str],
            batch_size: int,
            prefix: Optional[str] = "",
            **generation_params,
    ) -> List[List[Any]]:
        """Correct multiple sentences"""

    def train(self, config_path: str):
        with open(config_path) as infile:
            config = yaml.safe_load(infile)
        accelerator = Accelerator(mixed_precision=config['mixed_precision'],
                                  log_with=config['tracker_name'],
                                  project_dir=config['logging_path'],
                                  gradient_accumulation_steps=config['gradient_accumulation_steps'],
                                  split_batches=True)
        accelerator.init_trackers(
            project_name="spell_pretraining",
            config={'max_length': config['dataset']['max_length'],
                    'num_training_epochs': config['num_training_epochs'],
                    'gradient_accumulation_steps': config['gradient_accumulation_steps'],
                    'batch_size': config['batch_size'],
                    'mixed_precision': config['mixed_precision'],
                    'padding': config['padding'],
                    'optim': config['optim'],
                    'weight_decay': config['weight_decay'],
                    'learning_rate': config['learning_rate'],
                    'scheduler': config['scheduler'],
                    'mode': config['mode'],
                    'checkpoint_path': config['checkpoint_path']
                    }
        )
        with accelerator.main_process_first():
            self.model = models_names[config['model_type']].from_pretrained(self.model_name_or_path)
            train_tokenized, valid_tokenized = get_tokenized_datasets(self.tokenizer, **config['dataset'])
        train_loader = DataLoader(train_tokenized,
                                  batch_size=config['batch_size'],
                                  shuffle=False,
                                  pin_memory=False,
                                  num_workers=config['dataset']['num_workers'],
                                  collate_fn=TextCollatorWithPadding(self.tokenizer, self.model))
        valid = config['valid']
        if valid_tokenized is not None:
            valid_loader = DataLoader(valid_tokenized,
                                      batch_size=config['batch_size'],
                                      shuffle=False,
                                      pin_memory=False,
                                      num_workers=config['dataset']['num_workers'],
                                      collate_fn=TextCollatorWithPadding(self.tokenizer, self.model))
        else:
            valid_loader = None
            valid = False

        trainer = SageTrainer(
            accelerator,
            self.model,
            self.tokenizer,
            optimizer_name=config['optim'],
            scheduler_type=config['scheduler'],
            train_loader=train_loader,
            valid_loader=valid_loader,
            metric=config['metric'],
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            num_training_epochs=config['num_training_epochs'],
            gradient_accumulation_steps=config['gradient_accumulation_steps'],
            is_valid=valid,
            save_steps=config['save_steps'],
            checkpoint_path=config['checkpoint_path'],
            mode=config['mode'],
            gen_params=config['gen_params']
        )

        trainer.fit()
