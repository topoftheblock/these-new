"""
Reading the MS MARCO release.

MS MARCO v2.1 ships in two shapes and this module normalises both to one
record per query.

Column-oriented JSON (the official ``dev_v2.1.json``)::

    {"query":   {"0": "...", "1": "..."},
     "query_id":{"0": 1102432, ...},
     "passages":{"0": [{"passage_text": "...", "is_selected": 0}, ...], ...},
     "answers": {"0": ["..."], ...},
     "wellFormedAnswers": {"0": [...], ...}}

JSON Lines, one record per line, same field names.

Both are read lazily so a multi-gigabyte split never has to fit in memory at
once, except for the column-oriented form, which is a single JSON object and
must be parsed whole.
"""

import json
import sys

#: fields carried through to the rest of the pipeline
FIELDS = ("query_id", "query", "query_type", "answers",
          "wellFormedAnswers", "passages")


def load_rows(path):
    """Yield one dict per query, whatever shape the file is in.

    Dispatches on the extension: ``.parquet`` goes to
    :func:`preprocessing.huggingface.iter_rows`, which also transposes the
    Hugging Face passages struct. JSON and JSONL are handled here.

    Parameters
    ----------
    path : str or Path
        ``dev_v2.1.json`` (column-oriented), a ``.jsonl`` file, or a ``.parquet``
        shard from the Hugging Face export.

    Yields
    ------
    dict
        Keys as in :data:`FIELDS`. Missing fields become ``[]`` or ``None``
        rather than raising, because MS MARCO omits ``wellFormedAnswers``
        on some rows.
    """
    if str(path).endswith(".parquet"):
        from .huggingface import iter_rows
        yield from iter_rows(path)
        return

    with open(path, encoding="utf-8") as fh:
        first = fh.read(1)
        fh.seek(0)
        if first == "{":
            yield from _load_column_oriented(fh, path)
        else:
            yield from _load_jsonl(fh)


def _load_column_oriented(fh, path):
    blob = json.load(fh)
    if "query" not in blob:
        sys.exit(f"error: {path} has no 'query' key. "
                 "Is this the MS MARCO QnA release rather than the ranking one?")
    for key in blob["query"]:
        yield {
            "query_id": blob.get("query_id", {}).get(key),
            "query": blob["query"][key],
            "query_type": blob.get("query_type", {}).get(key),
            "answers": blob.get("answers", {}).get(key) or [],
            "wellFormedAnswers": blob.get("wellFormedAnswers", {}).get(key) or [],
            "passages": blob.get("passages", {}).get(key) or [],
        }


def _load_jsonl(fh):
    for line in fh:
        if not line.strip():
            continue
        row = json.loads(line)
        yield {f: row.get(f) or ([] if f != "query_id" else None) for f in FIELDS}
