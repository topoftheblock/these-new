"""
Loading MS MARCO from the Hugging Face parquet export.

Source: https://huggingface.co/datasets/microsoft/ms_marco (config ``v2.1``).

The parquet files are read directly rather than through the ``datasets``
library. That avoids a heavy dependency, and on this machine ``datasets`` cannot
be imported anyway because the environment has a pydantic / pydantic-core
version conflict. ``pyarrow`` reads the same files with no such problem.

Schema difference worth knowing
-------------------------------

The parquet export does **not** store passages the way the raw JSON release
does. Instead of a list of passage objects::

    "passages": [{"passage_text": "...", "is_selected": 0}, ...]

it stores one struct of parallel arrays::

    passages: struct<is_selected:   list<int32>,
                     passage_text:  list<string>,
                     url:           list<string>>

:func:`iter_rows` transposes that back into the per-passage form the rest of the
pipeline expects, so :mod:`preprocessing.filters` needs no special case.

Splits in v2.1: train (7 shards), validation (1 shard), test (1 shard).
Validation holds 101,093 rows, of which 12,467 carry a non-empty
``wellFormedAnswers``.
"""

import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

PARQUET_URL = ("https://huggingface.co/api/datasets/microsoft/ms_marco"
               "/parquet/{version}/{split}/{shard}.parquet")

#: columns the pipeline needs; url is skipped to keep the download smaller
COLUMNS = ["query_id", "query", "query_type", "answers",
           "wellFormedAnswers", "passages"]


def download(dest, split="validation", version="v2.1", shard=0, force=False):
    """Fetch one parquet shard, skipping the download if it is already there.

    Returns the path. The validation shard is about 200 MB.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    url = PARQUET_URL.format(version=version, split=split, shard=shard)
    urllib.request.urlretrieve(url, dest)
    return dest


def iter_rows(path, columns=None):
    """Yield one dict per query, transposed to the per-passage form.

    The parquet is read in row-group batches, so a multi-gigabyte train shard
    does not have to be held in memory at once.
    """
    columns = columns or COLUMNS
    available = set(pq.ParquetFile(path).schema_arrow.names)
    columns = [c for c in columns if c in available]

    for batch in pq.ParquetFile(path).iter_batches(columns=columns):
        for row in batch.to_pylist():
            yield normalise(row)


def normalise(row):
    """Transpose the passages struct into a list of passage dicts."""
    passages = row.get("passages") or {}
    texts = passages.get("passage_text") or []
    selected = passages.get("is_selected") or []
    urls = passages.get("url") or []

    row["passages"] = [
        {"passage_text": text,
         "is_selected": selected[i] if i < len(selected) else 0,
         "url": urls[i] if i < len(urls) else None}
        for i, text in enumerate(texts)
    ]
    return row
