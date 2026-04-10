import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from .phonetic_normalizer import PhoneticNormalizer


def vocab_size(series: pd.Series) -> int:
    vocab = set()
    for text in series.fillna("").astype(str):
        vocab.update(text.split())
    return len(vocab)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate normalization impact on vocabulary")
    parser.add_argument("--input", required=True, help="Path to CSV with `text` column")
    parser.add_argument("--output_dir", default="results", help="Directory for result files")
    parser.add_argument("--sample_size", type=int, default=100, help="Random sample size for before/after")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a `text` column")

    normalizer = PhoneticNormalizer()
    df["normalized_text"] = df["text"].fillna("").astype(str).map(normalizer.normalize)

    v_before = vocab_size(df["text"].fillna("").astype(str).str.lower())
    v_after = vocab_size(df["normalized_text"])
    reduction_pct = ((v_before - v_after) / max(v_before, 1)) * 100

    changed = (df["text"].fillna("").astype(str).str.lower() != df["normalized_text"]).mean() * 100

    metrics = pd.DataFrame([
        {
            "vocab_before": v_before,
            "vocab_after": v_after,
            "vocab_reduction_pct": round(reduction_pct, 2),
            "changed_rows_pct": round(changed, 2),
            "rows": len(df),
        }
    ])
    metrics.to_csv(out_dir / "normalization_metrics.csv", index=False)

    sample_n = min(args.sample_size, len(df))
    sample = df[["text", "normalized_text"]].sample(n=sample_n, random_state=42) if sample_n > 0 else df[["text", "normalized_text"]]
    sample.to_csv(out_dir / "before_after_samples.csv", index=False)

    # Top mappings estimated from token diffs
    pair_counter = Counter()
    for raw, norm in zip(df["text"].fillna(""), df["normalized_text"].fillna("")):
        raw_tokens = str(raw).lower().split()
        norm_tokens = str(norm).split()
        for left, right in zip(raw_tokens, norm_tokens):
            if left != right:
                pair_counter[(left, right)] += 1

    top_mappings = pd.DataFrame(
        [{"variant": k[0], "canonical": k[1], "count": v} for k, v in pair_counter.most_common(50)]
    )
    top_mappings.to_csv(out_dir / "top_variant_mappings.csv", index=False)

    print("Saved:")
    print(f"- {out_dir / 'normalization_metrics.csv'}")
    print(f"- {out_dir / 'before_after_samples.csv'}")
    print(f"- {out_dir / 'top_variant_mappings.csv'}")


if __name__ == "__main__":
    main()
