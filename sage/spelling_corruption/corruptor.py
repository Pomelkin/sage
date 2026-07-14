"""API to available methods of spelling corruption.

Currently, three options are available: word- and char-level Augmentex and
Statistical-based spelling corruption (SBSC).

Examples:
    from configuration_corruptor import CharAugConfig

    config = CharAugConfig(min_aug=10, max_aug=50, unit_prob=0.5)
    corruptor = CharAugCorruptor.from_config(config)
    print(corruptor.corrupt(sentence))

    ...

    corruptor = SBSCCorruptor.from_default_config()
    print(corruptor.corrupt(sentence))
"""

import dataclasses
from dataclasses import asdict
from typing import ClassVar, Generic, List, Optional, Type, TypeVar, Union
from abc import ABCMeta, abstractmethod

from augmentex.base import BaseAug
from augmentex.char import CharAug
from augmentex.word import WordAug

from .sbsc.sbsc import StatisticBasedSpellingCorruption
from .configuration_corruptor import WordAugConfig, CharAugConfig, SBSCConfig

EngineT = TypeVar("EngineT")
AugEngineT = TypeVar("AugEngineT", bound=BaseAug)
CorruptorT = TypeVar("CorruptorT", bound="Corruptor")

AnyConfig = Union[WordAugConfig, CharAugConfig, SBSCConfig]


class Corruptor(Generic[EngineT], metaclass=ABCMeta):
    """Base class for all corruptors.

    Attributes:
        engine_cls (ClassVar[Type]): engine class, instantiated by
            `from_config`/`from_default_config`;
        engine (EngineT): configured engine instance, set by
            `from_config`/`from_default_config`;
        config (Dict[str, Any]): config for every particular corruption class;
    """

    engine_cls: ClassVar[Type]
    engine: EngineT

    def __init__(self):
        self.config = asdict(self.get_default_config())

    @classmethod
    def from_config(cls: Type[CorruptorT], config: AnyConfig) -> CorruptorT:
        """Initialize corruptor from a given config.

        Args:
            config (Union[WordAugConfig, CharAugConfig, SBSCConfig]):
                config for every particular corruption class;

        Returns:
            particular corruptor class initialized from a given config;
        """
        corruptor = cls()
        corruptor.config = {
            field.name: getattr(config, field.name)
            for field in dataclasses.fields(config)
        }
        corruptor.engine = cls.engine_cls(**corruptor.config)

        return corruptor

    @classmethod
    def from_default_config(cls: Type[CorruptorT]) -> CorruptorT:
        """Initialize corruptor from a default config.

        Returns:
            particular corruptor class initialized from a default config;
        """
        corruptor = cls()
        corruptor.engine = cls.engine_cls(**corruptor.config)

        return corruptor

    @abstractmethod
    def corrupt(self, sentence: str, action: Optional[str] = None) -> str:
        pass

    @abstractmethod
    def batch_corrupt(
        self,
        sentences: List[str],
        action: Optional[str] = None,
        batch_prob: float = 0.3,
    ) -> List[str]:
        pass

    @staticmethod
    @abstractmethod
    def get_default_config() -> AnyConfig:
        pass


class AugCorruptor(Corruptor[AugEngineT], metaclass=ABCMeta):
    """Base class for Augmentex-based corruptors."""

    def corrupt(self, sentence: str, action: Optional[str] = None) -> str:
        return self.engine.augment(sentence, action=action)

    def batch_corrupt(
        self,
        sentences: List[str],
        action: Optional[str] = None,
        batch_prob: float = 0.3,
    ) -> List[str]:
        return self.engine.aug_batch(sentences, batch_prob=batch_prob, action=action)


class WordAugCorruptor(AugCorruptor[WordAug]):
    engine_cls: ClassVar[Type[WordAug]] = WordAug

    @staticmethod
    def get_default_config() -> WordAugConfig:
        return WordAugConfig()


class CharAugCorruptor(AugCorruptor[CharAug]):
    engine_cls: ClassVar[Type[CharAug]] = CharAug

    @staticmethod
    def get_default_config() -> CharAugConfig:
        return CharAugConfig()


class SBSCCorruptor(Corruptor[StatisticBasedSpellingCorruption]):
    engine_cls: ClassVar[Type[StatisticBasedSpellingCorruption]] = StatisticBasedSpellingCorruption

    def corrupt(self, sentence: str, action: Optional[str] = None, seed: Optional[int] = None) -> str:
        """Corrupt a sentence.

        Args:
            sentence (str): original sentence;
            action (Optional[str]): ignored, kept for interface compatibility;
            seed (Optional[int]): when None (default), the internal generator
                is used and advanced, so consecutive calls yield different
                corruptions while the whole sequence stays reproducible via
                config's :random_seed:; when given, the call is deterministic
                and does not affect the internal generator;
        """
        return self.engine.corrupt(sentence, seed=seed)

    def batch_corrupt(
        self,
        sentences: List[str],
        action: Optional[str] = None,
        batch_prob: float = 0.3,
        seed: Optional[int] = None,
    ) -> List[str]:
        """Corrupt a batch of sentences. See `corrupt` for :seed: semantics."""
        return self.engine.batch_corrupt(sentences, seed=seed)

    def reseed(self, seed: Optional[int] = None) -> None:
        """Reset the internal random generator (e.g. per DataLoader worker)."""
        self.engine.reseed(seed)

    @staticmethod
    def get_default_config() -> SBSCConfig:
        return SBSCConfig()
