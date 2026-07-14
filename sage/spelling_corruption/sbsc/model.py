"""
This module provides the main functionality to make statistical mistypings
that is embodied in Model class.

"""

import math
from typing import List, Dict, Optional, Tuple, Union

import numpy as np

from .base_classes import Fabric, Distribution
from .labeler import TyposTypes, PUNCTUATION_PATTERN
from ...utils.lang_utils import SUBSTITUTION_OPTIONS, AVAILABLE_LANG_CODES


class Model:
    """Statistical model parametrized by fetched distributions.

    Given parallel corpus, number of typos per sentence, types of error and their
    corresponding positions and substitution statistics are first gathered.
    Raw statistics are then fed to `Model` and normalized to appropriate discrete
    distributions. `Model` is parametrized by these distributions, and is used
    to corrupt text in a statistic-based manner.

    Attributes:
        debug_mode (bool): used for tests purposes;
        stats (Dict[str, List[int]]): used for tests purposes;
        lang (str): language of original text;
        skip_if_position_not_found (bool): whether to skip typo, when appropriate position cannot be found;

    Usage:
        from labeler import process_mistypings

        sources, corrections = load_data(...)
        typos_cnt, cm, stats = process_mistypings(sources, corrections)
        model = Model(typos_cnt, stats, cm, True, "ru")
        print(model.transform(clean_sentence))
    """

    names = [typo_type.name for typo_type in TyposTypes]

    # Registered dynamically in __init__ via `_register`.
    number_of_errors_per_sent: Distribution
    type_of_typo: Distribution

    def __init__(
        self,
        typos_count: List[int],
        stats: Dict[str, Dict[str, List[float]]],
        confusion_matrix: Dict[str, Dict[str, int]],
        skip_if_position_not_found: bool,
        lang: str,
        debug_mode: bool = False,
        random_seed: int = 42,
        min_typos: int = 1,
        max_typos: Optional[int] = None,
        scale_typos_by_length: bool = True,
        corpus_median_len: Optional[float] = None,
    ):
        # For debugging purposes only
        self.debug_mode = debug_mode
        self.stats = {
            "used_positions_pre": [],
            "used_positions_after": [],
            "pos": [],
        }

        if min_typos < 0:
            raise ValueError(
                "Provide non-negative :min_typos:, you provided {}".format(min_typos)
            )
        if max_typos is not None and max_typos < min_typos:
            raise ValueError(
                ":max_typos: ({}) must be >= :min_typos: ({})".format(
                    max_typos, min_typos
                )
            )
        if scale_typos_by_length and not (corpus_median_len and corpus_median_len > 0):
            raise ValueError(
                ":scale_typos_by_length: requires a positive :corpus_median_len:, you provided {}".format(
                    corpus_median_len
                )
            )
        self.min_typos = min_typos
        self.max_typos = max_typos
        self.scale_typos_by_length = scale_typos_by_length
        self.corpus_median_len = corpus_median_len

        self.rng = np.random.default_rng(random_seed)
        self.validate_inputs(stats, confusion_matrix, typos_count, lang)

        self.lang = lang.strip("_ ").lower()
        self.skip_if_position_not_found = skip_if_position_not_found

        # Number of mistypings per sentence, truncated to [min_typos, max_typos]:
        # out-of-range counts are dropped rather than clamped, so the empirical
        # distribution keeps its corpus shape conditioned on the allowed range
        # instead of piling the probability mass onto the bounds.
        bounded_typos_count = [
            count
            for count in typos_count
            if count >= min_typos and (max_typos is None or count <= max_typos)
        ]
        if not bounded_typos_count:
            raise ValueError(
                ":min_typos:/:max_typos: ({}/{}) drop every count of the reference corpus, "
                "whose counts range from {} to {}".format(
                    min_typos, max_typos, min(typos_count), max(typos_count)
                )
            )
        self._register_distribution(
            "number_of_errors_per_sent", bounded_typos_count, False
        )

        # Type of mistypings
        typos_cnt = {typo: len(v["abs"]) for typo, v in stats.items()}
        self._register("type_of_typo", Distribution.from_counts(typos_cnt))

        # Relative positions of mistypings
        self._bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for typo, v in stats.items():
            # To avoid 1.s being thrown in 11th bucket
            rel_positions = [pos if pos < 1.0 else pos - 0.00001 for pos in v["rel"]]

            buckets = np.digitize(rel_positions, self._bins)
            self._register_distribution(typo + "_positions", buckets)

        # Substitutions (confusion matrix)
        self._substitutions = {}
        for ch, candidates in confusion_matrix.items():
            substitutions = Distribution.from_counts(candidates)
            self._register("substitutions_for_{}".format(ord(ch)), substitutions)
            self._substitutions[ord(ch)] = substitutions

        # Options for characters missing from the confusion matrix, built once
        # instead of on every failed lookup in `transform`.
        self._fallback_substitutions = Distribution(
            getattr(SUBSTITUTION_OPTIONS, self.lang), False
        )

    @classmethod
    def validate_inputs(
        cls,
        stats: Dict[str, Dict[str, List[float]]],
        confusion_matrix: Dict[str, Dict[str, int]],
        typos_counts: List[int],
        lang: str,
    ):
        lang = lang.strip("_ ").lower()
        if lang not in AVAILABLE_LANG_CODES:
            raise ValueError(
                "Wrong language code: {}. Available codes are {}".format(
                    lang, " ".join(AVAILABLE_LANG_CODES)
                )
            )
        if len(stats) == 0:
            raise ValueError("Stats are empty, you should provide some")
        total_pos_num = 0
        for k, v in stats.items():
            if k not in cls.names:
                raise ValueError(
                    "You provided stats in wrong format, the key {} is not expected".format(
                        k
                    )
                )
            if len(v["abs"]) != len(v["rel"]):
                raise ValueError(
                    "Your inputs' lengths in stats (abs / rel) do not match for {}".format(
                        k
                    )
                )
            illegal_positions = [i for i, elem in enumerate(v["abs"]) if elem < 0]
            if len(illegal_positions) != 0:
                raise ValueError(
                    "Provide non-negative values for absolute positions for {} at positions {}".format(
                        k, illegal_positions
                    )
                )
            illegal_positions = [
                i for i, elem in enumerate(v["rel"]) if elem < 0 or elem > 1
            ]
            if len(illegal_positions) != 0:
                raise ValueError(
                    "Provide values between 0 and 1 for relative positions for {} at positions {}".format(
                        k, illegal_positions
                    )
                )
            total_pos_num += len(v["abs"])
        if total_pos_num == 0:
            raise ValueError("Provide some actual statistics")
        if len(typos_counts) == 0:
            raise ValueError("Typos counts are empty, you should provide some")
        if min(typos_counts) < 0:
            raise ValueError("Provide non-negative number of errors")
        if (
            len(confusion_matrix) == 0
            and "substitution" in stats
            and len(stats["substitution"]["abs"]) > 0
        ):
            raise ValueError("Confusion matrix is empty, but substitution is in stats")
        for k, v in confusion_matrix.items():
            if len(k) != 1:
                raise ValueError("Wrong format of key {} in confusion matrix".format(k))
            for sub, count in v.items():
                if len(sub) != 1:
                    raise ValueError(
                        "Wrong format of substitution {} in confusion matrix".format(
                            sub
                        )
                    )
                if count < 0:
                    raise ValueError(
                        "Provide non-negative value for count for key {} and substitution {}".format(
                            k, sub
                        )
                    )

    def _register_distribution(
        self,
        distribution: str,
        evidences: Union[List[int], np.ndarray],
        exclude_zero: bool = False,
    ):
        self._register(distribution, Distribution(evidences, exclude_zero))

    def _register(self, distribution: str, d: Distribution):
        if hasattr(self, distribution):
            raise ValueError(
                "You already defined that distribution {}".format(distribution)
            )
        setattr(self, distribution, d)

    def _factorization_scheme(
        self, interval_idx: int, sequence_length: int
    ) -> Tuple[int, int]:
        """Calculates exact absolute edge positions in a sentence, considering relative positions in a sentence.

        Args:
            interval_idx (int):
                interval id ranging from 1 to 10, representing equal non-overlaping semi-open
                intervals in [0,1];
            sequence_length (int): number of characters in a sentence;
        """
        left, right = self._bins[interval_idx - 1], self._bins[interval_idx]
        most_left = math.ceil(sequence_length * left)
        most_right = math.ceil(sequence_length * right)
        return most_left, most_right

    def _sample_num_typos(self, sentence: str, rng: np.random.Generator) -> int:
        """Sample the number of typos to insert into `sentence`.

        The base draw comes from the truncated-to-[min_typos, max_typos]
        empirical distribution, so it needs no further bounding. When
        :scale_typos_by_length: is on, the draw is rescaled by the sentence
        length relative to the corpus median and can leave the range again,
        so the bounds are re-applied as hard clamps. The result is always
        capped by the sentence length.
        """
        num_typos = int(self.number_of_errors_per_sent.sample(rng))
        if self.scale_typos_by_length:
            assert self.corpus_median_len is not None  # enforced in __init__
            # The corpus median was computed on labeler-preprocessed text, so
            # measure the input the same way to keep the ratio consistent.
            effective_len = len(PUNCTUATION_PATTERN.sub("", sentence.lower().strip()))
            scaled = num_typos * effective_len / self.corpus_median_len
            # Probabilistic rounding keeps the expected count equal to `scaled`.
            base = int(scaled)
            num_typos = base + int(rng.random() < scaled - base)
            num_typos = max(num_typos, self.min_typos)
            if self.max_typos is not None:
                num_typos = min(num_typos, self.max_typos)
        return min(num_typos, len(sentence))

    def transform(
        self, sentence: str, rng: Optional[Union[int, np.random.Generator]] = None
    ):
        """Spelling corruption procedure.

        The algorithm follows consequtive steps:
            1. Sample number of errors;
            2. For each error sample its type and corresponding interval in a sentence;
            3. Calculate absolute start and ending positions for typo;
            4. In a given interval find appropriate position for typo;
            5. Insert typo;

        Args:
            sentence (str): original sentence;
            rng (Union[int, np.random.Generator]): optional random generator or integer seed;
                defaults to the model's own generator, whose state advances with
                every call, so consecutive calls yield different corruptions
                while the whole sequence stays reproducible via `random_seed`;

        Returns:
            sentence (str): original sentence, but with errors;
        """
        if rng is None:
            rng = self.rng
        elif not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)

        # Sample number of mistypings
        num_typos = self._sample_num_typos(sentence, rng)
        fabric = Fabric()

        for _ in range(num_typos):
            # take len() for every typo, because with each
            # typo length of the sentence changes
            seqlen = len(sentence)

            # sample typo and corresponding interval for position
            typo = self.type_of_typo.sample(rng)
            handler = fabric.get_handler(typo)
            position_distribution = getattr(self, typo + "_positions")

            # sample bin a.k.a. interval for typo's position
            # and initial exact position inside this interval
            effective_tries = seqlen
            most_left, most_right = -1, -1
            while effective_tries >= 0:
                interval_idx = position_distribution.sample(rng)
                most_left, most_right = self._factorization_scheme(interval_idx, seqlen)
                if (
                    most_right - most_left >= 1
                ):  # for fixed bins that means length of sentence < 10
                    break
                effective_tries -= 1
            if most_right - most_left < 1:
                continue

            pos = rng.integers(low=most_left, high=most_right, size=1)[0]

            # Correct the position
            pos = handler.adjust_position(
                pos,
                most_left,
                most_right,
                self.skip_if_position_not_found,
                fabric.used_positions,
                rng,
                self.lang,
                sentence,
            )
            if pos is not None:
                substitutions = self._substitutions.get(
                    ord(sentence[pos].lower()), self._fallback_substitutions
                )
                sentence = handler.apply(pos, sentence, self.lang, rng, substitutions)

                if self.debug_mode:
                    used_positions_cp = fabric.used_positions.copy()
                    self.stats["used_positions_pre"].append(used_positions_cp)
                    self.stats["pos"].append(pos)

                fabric.finish(pos, typo)

                if self.debug_mode:
                    used_positions_cp = fabric.used_positions.copy()
                    self.stats["used_positions_after"].append(used_positions_cp)

        return sentence
