from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold


REQUIRED_COLUMNS = {"ID", "case", "target"}


def _normalize_case_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "case" not in df.columns and "casse" in df.columns:
        df = df.rename(columns={"casse": "case"})
    return df


def _load_split_csv(csv_path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = _normalize_case_column(df)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in CSV: {missing_str}")

    return df.copy()


def create_folds_by_case(
    data_path,
    output_path,
    n_splits=4,
    random_state=None,
):
    """
    Create case-level cross-validation folds using GroupKFold.

    Parameters
    ----------
    data_path : str or Path
        Path to the input CSV file.
    output_path : str or Path
        Path to save the output CSV file with a new 'fold' column.
    n_splits : int, default=4
        Number of folds.
    random_state : int or None, default=None
        Reserved for compatibility with external scripts. This argument is not
        used because GroupKFold is deterministic in this implementation.

    Returns
    -------
    pandas.DataFrame
        Input table with an added 'fold' column.
    """
    data_path = Path(data_path)
    output_path = Path(output_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {data_path}")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    data = pd.read_csv(data_path)
    data = _normalize_case_column(data)

    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in CSV: {missing_str}")

    n_cases = data["case"].nunique()
    if n_splits > n_cases:
        raise ValueError(
            f"n_splits ({n_splits}) cannot be greater than the number of unique "
            f"cases ({n_cases})."
        )

    group_kfold = GroupKFold(n_splits=n_splits)
    data = data.copy()
    data["fold"] = -1

    for fold, (_, val_idx) in enumerate(
        group_kfold.split(X=data, y=data["target"], groups=data["case"])
    ):
        data.loc[val_idx, "fold"] = fold

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Folds saved to {output_path}")

    return data


def create_folds_from_fixed_blocks(
    csv_paths,
    output_path,
):
    """
    Create folds by concatenating pre-defined CSV blocks.

    Example
    -------
    csv_paths = [test2.csv, test3.csv, test4.csv, test5.csv]

    Then:
      test2 -> fold 0
      test3 -> fold 1
      test4 -> fold 2
      test5 -> fold 3
    """
    output_path = Path(output_path)

    if csv_paths is None or len(csv_paths) < 2:
        raise ValueError("csv_paths must contain at least 2 CSV files.")

    frames = []
    seen_ids = set()

    for fold, csv_path in enumerate(csv_paths):
        df = _load_split_csv(csv_path)

        dup_ids = set(df["ID"]).intersection(seen_ids)
        if dup_ids:
            preview = ", ".join(sorted(map(str, list(dup_ids)[:10])))
            raise ValueError(
                f"Duplicate IDs found across fixed fold CSVs. Examples: {preview}"
            )

        seen_ids.update(df["ID"].tolist())

        df = df.copy()
        df["fold"] = fold
        df["source_csv"] = str(Path(csv_path).resolve())
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Fixed folds saved to {output_path}")

    return data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create case-level cross-validation folds."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["groupkfold", "fixed_blocks"],
        default="groupkfold",
        help="Fold creation mode.",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Path to the input CSV file for groupkfold mode.",
    )
    parser.add_argument(
        "--input-csvs",
        nargs="*",
        default=None,
        help="List of CSV files for fixed_blocks mode.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Path to save the output CSV file with fold assignments.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=4,
        help="Number of folds for groupkfold mode.",
    )
    parser.add_argument(
        "--show-summary",
        action="store_true",
        help="Print a summary table of fold, case, and target distribution.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "groupkfold":
        if args.input_csv is None:
            raise ValueError("--input-csv is required when mode=groupkfold")
        folds = create_folds_by_case(
            data_path=args.input_csv,
            output_path=args.output_csv,
            n_splits=args.n_splits,
        )
    elif args.mode == "fixed_blocks":
        if not args.input_csvs:
            raise ValueError("--input-csvs is required when mode=fixed_blocks")
        folds = create_folds_from_fixed_blocks(
            csv_paths=args.input_csvs,
            output_path=args.output_csv,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    if args.show_summary:
        summary = folds.groupby(["fold", "case", "target"]).size().unstack(fill_value=0)
        print(summary)


if __name__ == "__main__":
    main()