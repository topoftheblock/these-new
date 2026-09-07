"""
Download MS MARCO and keep only the rows with a well-formed answer.

Fetches one parquet shard from the Hugging Face export, drops every row whose
``wellFormedAnswers`` is empty, and writes the survivors back out as parquet.

Why filter here rather than at sampling time: the well-formed answer is the
reference string correctness is measured against, so a row without one cannot be
scored at all. Filtering once, up front, means every later step operates on rows
that are usable, and the count of what survived is recorded rather than being an
invisible side effect of sampling.

Usage::

    python3 -m preprocessing.fetch                       # validation split
    python3 -m preprocessing.fetch --split train --shard 0
"""

import argparse

import pyarrow as pa
import pyarrow.parquet as pq

import config

from .huggingface import COLUMNS, download


def is_well_formed(value):
    """True when the row carries at least one non-empty well-formed answer.

    The Hugging Face export stores this as a proper list. The raw JSON release
    sometimes stores the literal string ``"[]"`` instead, so that case is
    handled too and the same function works for both.
    """
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() not in ("", "[]")
    return any(a and a.strip() for a in value)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="validation",
                    choices=["train", "validation", "test"])
    ap.add_argument("--version", default="v2.1")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="re-download")
    args = ap.parse_args(argv)

    raw = config.DATA / f"ms_marco_{args.version}_{args.split}.parquet"
    out = config.DATA / f"ms_marco_{args.version}_{args.split}_wellformed.parquet"

    print(f"source: microsoft/ms_marco {args.version} / {args.split} "
          f"shard {args.shard}")
    if raw.exists() and not args.force:
        print(f"  already downloaded: {raw}")
    else:
        print("  downloading, this is a few hundred megabytes")
    download(raw, split=args.split, version=args.version,
             shard=args.shard, force=args.force)

    table = pq.read_table(raw, columns=COLUMNS)
    keep = [i for i, v in enumerate(table.column("wellFormedAnswers").to_pylist())
            if is_well_formed(v)]
    filtered = table.take(pa.array(keep))
    pq.write_table(filtered, out)

    n_in, n_out = table.num_rows, filtered.num_rows
    print(f"\n  rows read              : {n_in}")
    print(f"  with well-formed answer: {n_out}  ({100 * n_out / n_in:.1f}%)")
    print(f"  dropped                : {n_in - n_out}")
    print(f"\nwrote {out}")
    print(f"      {out.stat().st_size / 1e6:.1f} MB")
    print("\nnext:")
    print(f"  python3 -m preprocessing.build --input {out}")


if __name__ == "__main__":
    main()
