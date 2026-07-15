"""Utils for loading datasets from hub.

The `ai-forever/spellcheck_benchmark` and `ai-forever/spellcheck_punctuation_benchmark`
repos are legacy script-based datasets, which `datasets>=4.0` refuses to load
("Dataset scripts are no longer supported"). The underlying data are plain
JSON-lines files though, so we download them directly from the hub (cached by
`huggingface_hub`) and read them with pandas, bypassing the dataset script.
"""

import enum
from typing import Optional, Union, List, Tuple

import pandas as pd
from huggingface_hub import hf_hub_download


class DatasetsAvailable(enum.Enum):
    """Datasets available"""

    MultidomainGold = "Multidomain gold dataset. For more see `ai-forever/spellcheck_punctuation_benchmark`."
    RUSpellRU = "Social media texts and blogs. For more see `ai-forever/spellcheck_punctuation_benchmark`."
    MedSpellchecker = (
        "Medical anamnesis. For more see `ai-forever/spellcheck_punctuation_benchmark`."
    )
    GitHubTypoCorpusRu = (
        "Github commits. For more see `ai-forever/spellcheck_punctuation_benchmark`."
    )

    MultidomainGold_orth = "Multidomain gold dataset orthography only. For more see `ai-forever/spellcheck_benchmark`."
    RUSpellRU_orth = "Social media texts and blogs orthography only. For more see `ai-forever/spellcheck_benchmark`."
    MedSpellchecker_orth = "Medical anamnesis orthography only. For more see `ai-forever/spellcheck_benchmark`."
    GitHubTypoCorpusRu_orth = "Github commits orthography only. For more see `ai-forever/spellcheck_benchmark`."


datasets_available = [dataset.name for dataset in DatasetsAvailable]

# Splits shipped in the benchmark repos, in the order the legacy dataset
# script exposed them (matters for the split=None concatenation).
_DATASET_SPLITS = {
    "MultidomainGold": ("train", "test"),
    "RUSpellRU": ("train", "test"),
    "MedSpellchecker": ("test",),
    "GitHubTypoCorpusRu": ("test",),
}

# Columns the legacy dataset script exposed; extra fields in the raw
# JSON-lines files are dropped to keep the output schema unchanged.
_FEATURES = ["source", "correction", "domain"]


def _load_split(repo_id: str, dataset_name: str, split: str) -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=repo_id,
        filename="data/{}/{}.json".format(dataset_name, split),
        repo_type="dataset",
    )
    frame = pd.read_json(path, lines=True)
    return frame[[column for column in _FEATURES if column in frame.columns]]


def load_available_dataset_from_hf(
    dataset_name: str, for_labeler: bool, split: Optional[str] = None
) -> Union[Tuple[List[str], List[str]], pd.DataFrame]:
    if dataset_name not in datasets_available:
        raise ValueError(
            "You provided wrong dataset name: {}\nAvailable datasets are: {}".format(
                dataset_name, *datasets_available
            )
        )
    source_collection = "spellcheck_punctuation_benchmark"
    if dataset_name[-4:] == "orth":
        source_collection = "spellcheck_benchmark"
        dataset_name = dataset_name[:-5]
    repo_id = "ai-forever/{}".format(source_collection)

    available_splits = _DATASET_SPLITS[dataset_name]
    if split is None:
        dataset = pd.concat(
            [_load_split(repo_id, dataset_name, s) for s in available_splits]
        ).reset_index(drop=True)
    else:
        if split not in available_splits:
            raise ValueError(
                "Dataset {} has no split '{}', available splits are: {}".format(
                    dataset_name, split, ", ".join(available_splits)
                )
            )
        dataset = _load_split(repo_id, dataset_name, split)
    if for_labeler:
        sources = dataset.source.values.tolist()
        corrections = dataset.correction.values.tolist()
        return sources, corrections
    return dataset
