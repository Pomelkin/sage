"""Configuration classes for corruption methods.

Currently, three options are maintained: word- and char-level Augmentex and SBSC (Statistic-based
spelling corruption).

Examples:
    from corruptor import WordAugCorruptor

    config = WordAugConfig()
    corruptor = WordAugCorruptor.from_config(config)

    ...

    from corruptor import SBSCCorruptor

    config = SBSCConfig(
        lang="rus",
        reference_dataset_name_or_path="RUSpellRU"
    )
    corruptor = SBSCCorruptor.from_config(config)
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional


@dataclass
class BaseConfig:
    lang: str = field(default="rus", metadata={"help": "Source language rus/eng"})

    random_seed: Optional[int] = field(
        default=42,
        metadata={"help": "The random state for the application of augmentations."},
    )


@dataclass
class WordAugConfig(BaseConfig):
    """Word-level Augmentex config.

    Attributes:
        min_aug (int): The minimum amount of augmentation. Defaults to 1.
        max_aug (int): The maximum amount of augmentation. Defaults to 5.
        unit_prob (float): Percentage of the phrase to which augmentations will be applied. Defaults to 0.3.
    """

    min_aug: Optional[int] = field(
        default=1,
        metadata={"help": "The minimum amount of augmentation. Defaults to 1."},
    )

    max_aug: Optional[int] = field(
        default=5,
        metadata={"help": "The maximum amount of augmentation. Defaults to 5."},
    )

    unit_prob: Optional[float] = field(
        default=0.3,
        metadata={
            "help": "Percentage of the phrase to which augmentations will be applied. Defaults to 0.3."
        },
    )


@dataclass
class CharAugConfig(WordAugConfig):
    """Char-level Augmentex config.

    Attributes:
        min_aug (int): The minimum amount of augmentation. Defaults to 1.
        max_aug (int): The maximum amount of augmentation. Defaults to 5.
        unit_prob (float): Percentage of the phrase to which augmentations will be applied. Defaults to 0.3.
        mult_num (int): Maximum repetitions of characters. Defaults to 5.
    """

    mult_num: Optional[int] = field(
        default=5,
        metadata={"help": "Maximum repetitions of characters. Defaults to 5."},
    )


@dataclass
class SBSCConfig(BaseConfig):
    """Config for statistic-based spelling corruption.

    The number of typos per sentence is always estimated from the reference
    corpus; bound it with :min_typos:/:max_typos: and optionally rescale it to
    the length of the corrupted sentence with :scale_typos_by_length:.

    Attributes:
        lang (str): source language;
        stats (Dict[str, Dict[str, List[float]]]):
            types of typos and their absolute and relative positions in a sentence;
        confusion_matrix (Dict[str, Dict[str, int]]): Candidate replacements with corresponding frequencies;
        skip_if_position_not_found (bool):
            Whether to search for suitable position in a sentence when position is not found in interval;
        reference_dataset_name_or_path (bool): Path to or name of reference dataset
        reference_dataset_split (str): Dataset split to use when acquiring statistics.
        use_stats_cache (bool): Whether to cache reference dataset statistics on disk
            ($HOME/.cache/sage, override with SAGE_CACHE_DIR) keyed by corpus content.
        min_typos (int): Lower bound on the number of typos per sentence: corpus
            counts below it are dropped from the empirical distribution (truncation,
            not clamping). The default of 1 guarantees every sentence gets corrupted.
        max_typos (Optional[int]): Upper bound on the number of typos per sentence;
            corpus counts above it are dropped likewise.
        scale_typos_by_length (bool): Rescale the sampled number of typos by
            len(sentence) / median sentence length of the reference corpus.
    """

    stats: Optional[Dict[str, Dict[str, List[float]]]] = field(
        default=None,
        metadata={
            "help": "Relative and absolute positions of errors of corresponding types"
        },
    )

    confusion_matrix: Optional[Dict[str, Dict[str, int]]] = field(
        default=None,
        metadata={"help": "Candidate replacements with corresponding frequencies"},
    )

    skip_if_position_not_found: bool = field(
        default=True,
        metadata={
            "help": "Whether to search for suitable position in a sentence when position is not found in interval"
        },
    )

    reference_dataset_name_or_path: Optional[Union[str, os.PathLike]] = field(
        default="RUSpellRU",
        metadata={"help": "Path to or name of reference dataset"},
    )

    reference_dataset_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use when acquiring statistics."},
    )

    use_stats_cache: bool = field(
        default=True,
        metadata={
            "help": "Whether to cache reference dataset statistics on disk "
            "($HOME/.cache/sage, override with SAGE_CACHE_DIR)."
        },
    )

    min_typos: int = field(
        default=1,
        metadata={
            "help": "Lower bound on the number of typos per sentence; corpus counts "
            "below it are dropped from the empirical distribution. "
            "The default of 1 guarantees every sentence gets corrupted."
        },
    )

    max_typos: Optional[int] = field(
        default=None,
        metadata={
            "help": "Upper bound on the number of typos per sentence; corpus "
            "counts above it are dropped from the empirical distribution."
        },
    )

    scale_typos_by_length: bool = field(
        default=True,
        metadata={
            "help": "Rescale the sampled number of typos by "
            "len(sentence) / median sentence length of the reference corpus."
        },
    )
