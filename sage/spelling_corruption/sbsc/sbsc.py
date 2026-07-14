"""API to Statistical-based Spelling Corruption method.

Examples:
    corruptor = StatisticBasedSpellingCorruption(
        lang="rus",
        reference_dataset_name_or_path="RUSpellRU",
    )
    # Consecutive calls advance the internal generator: each call gives a new
    # corruption, and the whole sequence is reproducible via :random_seed:.
    print(corruptor.corrupt(sentence))

    # A detached, deterministic one-off corruption:
    print(corruptor.corrupt(sentence, seed=1))

    ....

    from labeler import process_mistypings

    sources, corrections = load_data(...)
    typos_cnt, cm, stats = process_mistypings(sources, corrections)
    corruptor = StatisticBasedSpellingCorruption(
        lang="rus",
        typos_count=typos_cnt,
        stats=stats,
        confusion_matrix=cm,
    )
    print(corruptor.corrupt(sentence))
"""

import os
from typing import List, Dict, Optional, Union

import numpy as np
import pandas as pd

from . import stats_cache
from .model import Model
from .labeler import process_mistypings
from ...utils.data_load_utils import load_available_dataset_from_hf, DatasetsAvailable

datasets_available = [dataset.name for dataset in DatasetsAvailable]


def _labeler_stats(sources: List[str], corrections: List[str], use_cache: bool):
    """process_mistypings with a best-effort disk cache keyed by a corpus content hash."""
    if not use_cache:
        return process_mistypings(sources, corrections)
    key = stats_cache.corpus_hash(sources, corrections)
    cached = stats_cache.load(key)
    if cached is not None:
        return cached
    stats, confusion_matrix, typos_count = process_mistypings(sources, corrections)
    stats_cache.save(key, stats, confusion_matrix, typos_count)
    return stats, confusion_matrix, typos_count


class StatisticBasedSpellingCorruption:
    """API to `Model` class from model.py.

    Attributes:
        model (model.Model): statistic-based spelling corruption model;
    """

    def __init__(
        self,
        lang: str,
        typos_count: Optional[List[int]] = None,
        stats: Optional[Dict[str, Dict[str, List[float]]]] = None,
        confusion_matrix: Optional[Dict[str, Dict[str, int]]] = None,
        skip_if_position_not_found: bool = True,
        reference_dataset_name_or_path: Optional[Union[str, os.PathLike]] = None,
        reference_dataset_split: str = "train",
        random_seed: Optional[int] = 42,
        use_stats_cache: bool = True,
    ):
        typos_count_ = None
        stats_ = None
        confusion_matrix_ = None

        if (
            typos_count is None or stats is None or confusion_matrix is None
        ) and reference_dataset_name_or_path is None:
            raise RuntimeError("""You should provide at least one of :typos_count:/:stats:/:confusion_matrix:
                                or :reference_dataset_name_or_path:""")
        if (
            typos_count is None or stats is None or confusion_matrix is None
        ) and reference_dataset_name_or_path is not None:
            reference_dataset_name_or_path = str(reference_dataset_name_or_path)
            if reference_dataset_name_or_path in datasets_available:
                sources, corrections = load_available_dataset_from_hf(
                    reference_dataset_name_or_path,
                    for_labeler=True,
                    split=reference_dataset_split,
                )
                stats_, confusion_matrix_, typos_count_ = _labeler_stats(
                    sources, corrections, use_stats_cache
                )
            elif os.path.isdir(reference_dataset_name_or_path):
                if os.path.isfile(
                    os.path.join(reference_dataset_name_or_path, "sources.txt")
                ) and os.path.isfile(
                    os.path.join(reference_dataset_name_or_path, "corrections.txt")
                ):
                    src_file = open(
                        os.path.join(reference_dataset_name_or_path, "sources.txt"),
                        encoding="utf8",
                    )
                    corr_file = open(
                        os.path.join(reference_dataset_name_or_path, "corrections.txt"),
                        encoding="utf8",
                    )
                    sources = src_file.read().split("\n")
                    corrections = corr_file.read().split("\n")
                    src_file.close()
                    corr_file.close()
                    if len(sources) != len(corrections):
                        raise RuntimeError(
                            "Sources and corrections must be of the same length, but get {} vs {}".format(
                                len(sources), len(corrections)
                            )
                        )
                    stats_, confusion_matrix_, typos_count_ = _labeler_stats(
                        sources, corrections, use_stats_cache
                    )
                elif os.path.isfile(
                    os.path.join(reference_dataset_name_or_path, "data.csv")
                ):
                    try:
                        data = pd.read_csv(
                            os.path.join(reference_dataset_name_or_path, "data.csv")
                        )
                    except Exception as e:
                        raise RuntimeError(
                            "Wrong format of file {}. Raised an error: {}".format(
                                os.path.join(
                                    reference_dataset_name_or_path, "data.csv"
                                ),
                                str(e),
                            )
                        )
                    if not ("source" in data and "correction" in data):
                        raise RuntimeError(
                            "You must provide 'source' and 'correction' columns in {}".format(
                                os.path.join(reference_dataset_name_or_path, "data.csv")
                            )
                        )
                    if data.isna().any().max():  # ty:ignore[unresolved-attribute]
                        raise ValueError(
                            "Your data at {} contain unnecessary nans".format(
                                os.path.join(reference_dataset_name_or_path, "data.csv")
                            )
                        )
                    sources = data.source.values.tolist()
                    corrections = data.correction.values.tolist()
                    stats_, confusion_matrix_, typos_count_ = _labeler_stats(
                        sources, corrections, use_stats_cache
                    )
                else:
                    raise RuntimeError(
                        "You must provide either 'data.csv' or 'sources.txt'/'corrections.txt' in {}".format(
                            reference_dataset_name_or_path
                        )
                    )
            else:
                raise ValueError(
                    "You must provide either valid path or available dataset's name, you provided {}".format(
                        reference_dataset_name_or_path
                    )
                )
        if typos_count is not None:
            typos_count_ = typos_count
        if stats is not None:
            stats_ = stats
        if confusion_matrix is not None:
            confusion_matrix_ = confusion_matrix

        self.model = Model(
            typos_count=typos_count_,  # ty:ignore[invalid-argument-type]
            stats=stats_,  # ty:ignore[invalid-argument-type]
            confusion_matrix=confusion_matrix_,  # ty:ignore[invalid-argument-type]
            skip_if_position_not_found=skip_if_position_not_found,
            lang=lang,
            random_seed=random_seed,  # ty:ignore[invalid-argument-type]
        )

    @staticmethod
    def show_reference_datasets_available():
        print(*datasets_available, sep="\n")

    def reseed(self, seed: Optional[int] = None) -> None:
        """Reset the internal random generator.

        Useful for e.g. DataLoader workers, which inherit an identical copy of
        the corruptor and would otherwise all produce the same random stream.
        """
        self.model.rng = np.random.default_rng(seed)

    def corrupt(self, sentence: str, seed: Optional[int] = None) -> str:
        """Corrupt a sentence.

        Args:
            sentence (str): original sentence;
            seed (Optional[int]): when None (default), the internal generator
                is used and advanced, so consecutive calls yield different
                corruptions while the whole sequence stays reproducible via
                :random_seed:; when given, the call is deterministic and does
                not affect the internal generator;
        """
        rng = self.model.rng if seed is None else np.random.default_rng(seed)
        return self.model.transform(sentence, rng)

    def batch_corrupt(self, sentences: List[str], seed: Optional[int] = None) -> List[str]:
        """Corrupt a batch of sentences. See `corrupt` for :seed: semantics."""
        rng = self.model.rng if seed is None else np.random.default_rng(seed)
        return [self.model.transform(sentence, rng) for sentence in sentences]
