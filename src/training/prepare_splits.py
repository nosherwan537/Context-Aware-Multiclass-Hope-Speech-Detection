import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Create stratified train/val/test splits")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output_dir", default="data/splits", help="Output split directory")
    parser.add_argument("--label_col", default="label", help="Label column name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required = {"text", args.label_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "source" not in df.columns:
        df["source"] = "unknown"

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=args.seed,
        stratify=df[args.label_col],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=args.seed,
        stratify=temp_df[args.label_col],
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cols = ["text", args.label_col, "source"]
    train_df[cols].rename(columns={args.label_col: "label"}).to_csv(out / "train.csv", index=False)
    val_df[cols].rename(columns={args.label_col: "label"}).to_csv(out / "val.csv", index=False)
    test_df[cols].rename(columns={args.label_col: "label"}).to_csv(out / "test.csv", index=False)

    summary = pd.DataFrame(
        [
            {"split": "train", "rows": len(train_df)},
            {"split": "val", "rows": len(val_df)},
            {"split": "test", "rows": len(test_df)},
        ]
    )
    summary.to_csv(out / "split_summary.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
