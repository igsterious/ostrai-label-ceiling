"""Loading and saving the benchmark TSVs.

Every read and write in this project goes through here so the parsing
settings stay identical everywhere. The important one is quoting=QUOTE_NONE.

Why that matters: pandas defaults to treating " as a quote character, so a
message like  He said "hello" to me  can get its fields merged or split in
ways that are silent -- you get a dataframe, it just has wrong text in it.
The provided files happen to contain zero quote characters (verified), but
anything we generate later might, so we disable quote handling on both sides
and treat the files as strictly tab-separated. Tabs inside text would break
this, and validate() below checks for exactly that.
"""

import csv

import pandas as pd

from config import benchmark_dir

# Applied to every read: no quote character, so " is just a character.
_READ_KWARGS = dict(sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)


def _read_tsv(path, expected_columns):
    df = pd.read_csv(path, **_READ_KWARGS)
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing column(s) {missing}; found {list(df.columns)}")
    return df


def write_tsv(df, path):
    """Counterpart to _read_tsv. Raises if any cell contains a tab or newline,
    since that would produce a file we cannot read back correctly."""
    for column in df.columns:
        bad = df[column].astype(str).str.contains(r"[\t\r\n]", regex=True)
        if bad.any():
            raise ValueError(
                f"cannot write {path}: column {column!r} has {bad.sum()} value(s) "
                "containing a tab or newline, which would corrupt the TSV"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONE)
    return path


def load_intents(benchmark):
    """The label space, as an ordered list of category names."""
    path = benchmark_dir(benchmark) / "intents.txt"
    intents = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [i for i in intents if i]


def load_labeled(benchmark):
    """The 10-per-class ground truth. Columns: id, text, intent."""
    return _read_tsv(benchmark_dir(benchmark) / "labeled_examples.tsv", ["id", "text", "intent"])


def load_pool(benchmark):
    """Unlabeled messages available for pseudo-labelling. Columns: id, text.

    Note this INCLUDES the labeled examples -- they are a subset of the pool,
    by both id and text. Use load_pool_unlabeled() to get just the rest.
    """
    return _read_tsv(benchmark_dir(benchmark) / "pool.tsv", ["id", "text"])


def load_pool_unlabeled(benchmark):
    """The pool minus the rows we already have gold labels for."""
    pool = load_pool(benchmark)
    known_ids = set(load_labeled(benchmark)["id"])
    return pool[~pool["id"].isin(known_ids)].reset_index(drop=True)


def load_test(benchmark):
    """Scoring data. Columns: id, text.

    HARD RULE 1: this is for producing final predictions only. Never fit,
    tune, or select anything on it, and never send it to the labeller.
    """
    return _read_tsv(benchmark_dir(benchmark) / "test.tsv", ["id", "text"])


def validate(benchmark):
    """Sanity-check one benchmark's files and return a summary dict.

    Cheap to run and worth running after any change to the data on disk.
    """
    intents = load_intents(benchmark)
    labeled = load_labeled(benchmark)
    pool = load_pool(benchmark)
    test = load_test(benchmark)

    per_class = labeled["intent"].value_counts()
    unknown = sorted(set(labeled["intent"]) - set(intents))
    if unknown:
        raise ValueError(f"{benchmark}: labeled data uses intents not in intents.txt: {unknown}")
    uncovered = sorted(set(intents) - set(labeled["intent"]))
    if uncovered:
        raise ValueError(f"{benchmark}: intents with no labeled examples: {uncovered}")

    for name, df in (("pool", pool), ("test", test)):
        if df["id"].duplicated().any():
            raise ValueError(f"{benchmark}: duplicate ids in {name}.tsv")

    return {
        "benchmark": benchmark,
        "n_intents": len(intents),
        "n_labeled": len(labeled),
        "per_class_min": int(per_class.min()),
        "per_class_max": int(per_class.max()),
        "n_pool": len(pool),
        "n_pool_unlabeled": len(load_pool_unlabeled(benchmark)),
        "n_test": len(test),
    }


if __name__ == "__main__":
    # Re-checks the data facts quoted in CLAUDE.md. Run after any change to
    # the files under data/.
    from config import BENCHMARKS

    for b in BENCHMARKS:
        s = validate(b)
        labeled_ids = set(load_labeled(b)["id"])
        pool_ids = set(load_pool(b)["id"])
        pool_text = set(load_pool(b)["text"])
        test_text = set(load_test(b)["text"])
        print(
            f"{s['benchmark']}: {s['n_intents']} intents, {s['n_labeled']} labeled "
            f"({s['per_class_min']}-{s['per_class_max']}/class), pool {s['n_pool']} "
            f"({s['n_pool_unlabeled']} unlabeled), test {s['n_test']}\n"
            f"  labeled is subset of pool: {labeled_ids <= pool_ids}\n"
            f"  pool/test text overlap: {len(pool_text & test_text)}"
        )
