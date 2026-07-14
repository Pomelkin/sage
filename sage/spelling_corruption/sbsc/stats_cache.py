"""Disk cache for reference-corpus statistics used by SBSC.

Gathering typos statistics with the labeler is quadratic in sentence length
and takes ~10 seconds per 2000 sentence pairs, which is paid on every
corruptor initialization. The statistics depend only on the corpus content,
so they are cached as JSON keyed by a sha256 hash of the corpus.

Cache location: $SAGE_CACHE_DIR/sbsc_stats (or ~/.cache/sage/sbsc_stats).
The cache is best-effort: any IO or format problem silently falls back to
recomputing the statistics.
"""

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

_CACHE_FORMAT = 1

StatsTuple = Tuple[Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, int]], List[int]]


def _cache_dir() -> str:
    root = os.environ.get("SAGE_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "sage")
    return os.path.join(root, "sbsc_stats")


def corpus_hash(sources: List[str], corrections: List[str]) -> str:
    """Content hash of a parallel corpus, independent of its origin."""
    h = hashlib.sha256()
    for sentence in sources:
        h.update(sentence.encode("utf8"))
        h.update(b"\x00")
    h.update(b"\x01")
    for sentence in corrections:
        h.update(sentence.encode("utf8"))
        h.update(b"\x00")
    return h.hexdigest()


def load(key: str) -> Optional[StatsTuple]:
    path = os.path.join(_cache_dir(), key + ".json")
    try:
        with open(path, encoding="utf8") as f:
            payload = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("format") != _CACHE_FORMAT:
        return None
    try:
        return payload["stats"], payload["confusion_matrix"], payload["typos_count"]
    except KeyError:
        return None


def save(key: str, stats: Dict, confusion_matrix: Dict, typos_count: List[int]) -> None:
    payload = {
        "format": _CACHE_FORMAT,
        "stats": stats,
        "confusion_matrix": confusion_matrix,
        "typos_count": typos_count,
    }
    directory = _cache_dir()
    tmp_path = os.path.join(directory, ".tmp-{}-{}.json".format(os.getpid(), key))
    try:
        os.makedirs(directory, exist_ok=True)
        with open(tmp_path, "w", encoding="utf8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_path, os.path.join(directory, key + ".json"))
    except OSError:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
