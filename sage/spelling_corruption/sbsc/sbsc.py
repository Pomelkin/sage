"""API to Statistical-based Spelling Corruption method.

Examples:
    corruptor = StatisticBasedSpellingCorruption(
        lang="rus",
        reference_dataset_name_or_path="RUSpellRU",
    )
    # Consecutive calls advance the internal generator: each call gives a new
    # corruption. With the default :random_seed:=None the generator is lazily
    # seeded from numpy's global random state on first use, so per-worker
    # seeding (e.g. a DataLoader `worker_init_fn`) is picked up; pass an
    # explicit :random_seed: to make the whole sequence reproducible.
    print(corruptor.corrupt(sentence))

    # A detached, deterministic one-off corruption:
    print(corruptor.corrupt(sentence, seed=1))

    ....

    # Guaranteed 1..3 errors per sentence, scaled to the length of the input
    # (relative to the median sentence length of the reference corpus):
    corruptor = StatisticBasedSpellingCorruption(
        lang="rus",
        reference_dataset_name_or_path="RUSpellRU",
        min_typos=1,
        max_typos=3,
        scale_typos_by_length=True,
    )
    print(corruptor.corrupt(sentence))
"""

import os
from statistics import median
from typing import List, Dict, Optional, Union

import numpy as np
import pandas as pd

from . import stats_cache
from .model import Model
from .labeler import process_mistypings, PUNCTUATION_PATTERN
from ...utils.data_load_utils import load_available_dataset_from_hf, DatasetsAvailable

datasets_available = [dataset.name for dataset in DatasetsAvailable]


def _compute_labeler_stats(sources: List[str], corrections: List[str]):
    stats, confusion_matrix, typos_count = process_mistypings(sources, corrections)
    # Median length of the processed reference sentences; the same
    # preprocessing the labeler applies before gathering statistics.
    processed_lens = [
        len(PUNCTUATION_PATTERN.sub("", sentence.lower().strip()))
        for sentence in sources
    ]
    corpus_median_len = float(median(processed_lens)) if processed_lens else 0.0
    return stats, confusion_matrix, typos_count, corpus_median_len


def _labeler_stats(sources: List[str], corrections: List[str], use_cache: bool):
    """_compute_labeler_stats with a best-effort disk cache keyed by a corpus content hash."""
    if not use_cache:
        return _compute_labeler_stats(sources, corrections)
    key = stats_cache.corpus_hash(sources, corrections)
    cached = stats_cache.load(key)
    if cached is not None:
        return cached
    stats, confusion_matrix, typos_count, corpus_median_len = _compute_labeler_stats(
        sources, corrections
    )
    stats_cache.save(key, stats, confusion_matrix, typos_count, corpus_median_len)
    return stats, confusion_matrix, typos_count, corpus_median_len


class StatisticBasedSpellingCorruption:
    """API to `Model` class from model.py.

    The number of typos per sentence is always estimated from the reference
    corpus (it cannot be passed manually): bound it with :min_typos:/:max_typos:
    (corpus counts outside the range are dropped, truncating the empirical
    distribution without distorting its shape) and optionally scale it to the
    length of the corrupted sentence with :scale_typos_by_length:. :stats: and
    :confusion_matrix: can still be overridden manually.

    Attributes:
        model (model.Model): statistic-based spelling corruption model;
    """

    def __init__(
        self,
        lang: str,
        stats: Optional[Dict[str, Dict[str, List[float]]]] = None,
        confusion_matrix: Optional[Dict[str, Dict[str, int]]] = None,
        skip_if_position_not_found: bool = True,
        reference_dataset_name_or_path: Optional[Union[str, os.PathLike]] = None,
        reference_dataset_split: str = "train",
        random_seed: Optional[int] = None,
        use_stats_cache: bool = True,
        min_typos: int = 1,
        max_typos: Optional[int] = None,
        scale_typos_by_length: bool = True,
    ):
        if reference_dataset_name_or_path is None:
            raise RuntimeError(
                "You should provide :reference_dataset_name_or_path:: the number of typos "
                "per sentence is always estimated from a reference corpus. Bound it via "
                ":min_typos:/:max_typos: instead of passing custom counts."
            )

        reference_dataset_name_or_path = str(reference_dataset_name_or_path)
        if reference_dataset_name_or_path in datasets_available:
            sources, corrections = load_available_dataset_from_hf(
                reference_dataset_name_or_path,
                for_labeler=True,
                split=reference_dataset_split,
            )
        elif os.path.isdir(reference_dataset_name_or_path):
            if os.path.isfile(
                os.path.join(reference_dataset_name_or_path, "sources.txt")
            ) and os.path.isfile(
                os.path.join(reference_dataset_name_or_path, "corrections.txt")
            ):
                with open(
                    os.path.join(reference_dataset_name_or_path, "sources.txt"),
                    encoding="utf8",
                ) as src_file:
                    sources = src_file.read().split("\n")
                with open(
                    os.path.join(reference_dataset_name_or_path, "corrections.txt"),
                    encoding="utf8",
                ) as corr_file:
                    corrections = corr_file.read().split("\n")
                if len(sources) != len(corrections):
                    raise RuntimeError(
                        "Sources and corrections must be of the same length, but get {} vs {}".format(
                            len(sources), len(corrections)
                        )
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
                            os.path.join(reference_dataset_name_or_path, "data.csv"),
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

        stats_, confusion_matrix_, typos_count_, corpus_median_len_ = _labeler_stats(
            sources, corrections, use_stats_cache
        )
        if stats is not None:
            stats_ = stats
        if confusion_matrix is not None:
            confusion_matrix_ = confusion_matrix

        self.model = Model(
            typos_count=typos_count_,
            stats=stats_,
            confusion_matrix=confusion_matrix_,
            skip_if_position_not_found=skip_if_position_not_found,
            lang=lang,
            random_seed=random_seed,
            min_typos=min_typos,
            max_typos=max_typos,
            scale_typos_by_length=scale_typos_by_length,
            corpus_median_len=corpus_median_len_,
        )

    @staticmethod
    def show_reference_datasets_available():
        print(*datasets_available, sep="\n")

    def reseed(self, seed: Optional[int] = None) -> None:
        """Reset the internal random generator.

        With an explicit :seed: the generator is reset deterministically.
        With :seed:=None the generator is dropped and will be lazily re-seeded
        from numpy's global random state on next use — handy in DataLoader
        workers, which inherit an identical copy of the corruptor and would
        otherwise all produce the same random stream.
        """
        self.model._rng = None if seed is None else np.random.default_rng(seed)

    def corrupt(self, sentence: str, seed: Optional[int] = None) -> str:
        """Corrupt a sentence.

        Args:
            sentence (str): original sentence;
            seed (Optional[int]): when None (default), the internal generator
                is used and advanced, so consecutive calls yield different
                corruptions (reproducible when :random_seed: was given at
                construction); when given, the call is deterministic and does
                not affect the internal generator;
        """
        rng = self.model.rng if seed is None else np.random.default_rng(seed)
        return self.model.transform(sentence, rng)

    def batch_corrupt(
        self, sentences: List[str], seed: Optional[int] = None
    ) -> List[str]:
        """Corrupt a batch of sentences. See `corrupt` for :seed: semantics."""
        rng = self.model.rng if seed is None else np.random.default_rng(seed)
        return [self.model.transform(sentence, rng) for sentence in sentences]
