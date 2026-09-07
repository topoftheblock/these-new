"""
Embed the corpus and build the vector database.

Takes every passage attached to a row with a well-formed answer, embeds it with
the retrieval encoder, and writes a FAISS index plus its passage-id map.

About 122,700 unique passages come out of the validation split. Their vectors
occupy 0.38 GB at float32, so the default index is **flat**: exhaustive search,
no approximation. Use ``--index ivfpq`` only if the corpus grows past what memory
allows, and record that the results then come from an approximate index.

The job is resumable. A checkpoint recording how many passages have been added
is written periodically, and ``--resume`` continues from there rather than
re-embedding what is already indexed.

Usage::

    python3 -m ingest.build_index --limit 2000     # trial run
    python3 -m ingest.build_index                  # the whole corpus
    python3 -m ingest.build_index --resume         # continue after a stop
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import config
from ingest.corpus_source import count, iter_passages
from ingest.vector_store import DEFAULT_M, DEFAULT_NLIST, FaissStore

DEFAULT_SOURCE = "ms_marco_v2.1_validation_wellformed.parquet"


def _checkpoint(store_dir):
    return Path(store_dir) / "checkpoint.json"


def read_checkpoint(store_dir):
    path = _checkpoint(store_dir)
    return json.loads(path.read_text())["n_added"] if path.exists() else 0


def write_checkpoint(store_dir, n_added, source):
    _checkpoint(store_dir).write_text(json.dumps(
        {"n_added": n_added, "source": str(source), "updated": time.time()},
        indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=None,
                    help=f"well-formed parquet (default: data/{DEFAULT_SOURCE})")
    ap.add_argument("--store", default=None,
                    help="output directory (default: data/vector_store)")
    ap.add_argument("--index", choices=["flat", "ivfpq"], default="flat")
    ap.add_argument("--limit", type=int, help="stop after N passages")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--checkpoint-every", type=int, default=40,
                    help="batches between checkpoint writes")
    ap.add_argument("--train-size", type=int, default=262_144,
                    help="ivfpq only: passages used to fit the index")
    ap.add_argument("--nlist", type=int, default=DEFAULT_NLIST)
    ap.add_argument("--m", type=int, default=DEFAULT_M)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    source = Path(args.source or config.DATA / DEFAULT_SOURCE)
    store_dir = Path(args.store or config.DATA / "vector_store")
    if not source.exists():
        raise SystemExit(f"{source} not found. Run preprocessing.fetch first.")

    from rag import Embedder
    embedder = Embedder()

    total = args.limit or count(source)
    print(f"source  : {source.name}")
    print(f"encoder : {embedder.model_id} on {embedder.device}, {embedder.dim} dims")
    print(f"index   : {args.index}")
    print(f"passages: {total:,}")

    start = read_checkpoint(store_dir) if args.resume else 0
    if args.resume and start:
        store = FaissStore.load(store_dir)
        print(f"resuming: {start:,} already indexed")
    elif args.index == "flat":
        store = FaissStore.flat(embedder.dim)
    else:
        store = FaissStore.ivfpq(embedder.dim, nlist=args.nlist, m=args.m)
        needed = max(args.train_size, store.min_training_vectors())
        if needed > total:
            raise SystemExit(f"ivfpq needs {needed:,} training vectors but only "
                             f"{total:,} passages are available; use --index flat")
        print(f"\ntraining on {needed:,} passages")
        sample, seen = [], 0
        for _, texts in iter_passages(source, batch_size=args.batch_size):
            sample.append(embedder.encode(texts))
            seen += len(texts)
            print(f"  {seen:,}/{needed:,}", end="\r")
            if seen >= needed:
                break
        store.train(np.vstack(sample)[:needed])
        del sample
        print("\n  fitted")

    print(f"\nembedding from passage {start:,}")
    t0, n_added, n_batches = time.time(), 0, 0

    for pids, texts in iter_passages(source, batch_size=args.batch_size):
        if n_added + len(pids) <= start:      # already indexed, skip
            n_added += len(pids)
            continue
        if n_added < start:                   # partial batch at the boundary
            offset = start - n_added
            pids, texts = pids[offset:], texts[offset:]
            n_added = start
        if n_added >= total:
            break
        if n_added + len(pids) > total:
            pids, texts = pids[:total - n_added], texts[:total - n_added]

        store.add(embedder.encode(texts), pids)
        n_added += len(pids)
        n_batches += 1

        if n_batches % args.checkpoint_every == 0:
            store.save(store_dir)
            write_checkpoint(store_dir, n_added, source)
            rate = (n_added - start) / max(time.time() - t0, 1e-9)
            eta = (total - n_added) / rate if rate else 0
            print(f"  {n_added:>8,}/{total:,}  {rate:5.0f}/s  "
                  f"eta {eta / 60:5.1f} min", end="\r")

    store.save(store_dir)
    write_checkpoint(store_dir, n_added, source)
    size = sum(f.stat().st_size for f in store_dir.iterdir() if f.is_file())
    print(f"\n\nindexed {len(store):,} passages in {(time.time() - t0) / 60:.1f} min")
    print(f"store: {size / 1e6:.0f} MB at {store_dir}  (exact={store.is_exact})")


if __name__ == "__main__":
    main()
