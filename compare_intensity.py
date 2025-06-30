import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_excel_scatter import _find_column


def _numeric_column(df: pd.DataFrame) -> str:
    col = _find_column(df, "Numeric_area")
    if col:
        return col
    c2h = _find_column(df, "Numeric_2H_area")
    c6h = _find_column(df, "Numeric_6H_area")
    if c2h and c6h:
        df["Numeric_area"] = df[c2h] + df[c6h]
        return "Numeric_area"
    raise KeyError("Numeric intensity columns not found")


def compute_metrics(df: pd.DataFrame) -> dict:
    total_col = _find_column(df, "Total_scaled")
    if not total_col:
        raise KeyError("Total_scaled column not found")
    num_col = _numeric_column(df)
    diff = df[num_col] - df[total_col]
    ratio = df[num_col] / df[total_col].replace(0, np.nan)
    rmse = float(np.sqrt(np.nanmean(diff ** 2)))
    mean_ratio = float(np.nanmean(ratio))
    return {"rmse": rmse, "mean_ratio": mean_ratio}


def plot_comparison(df: pd.DataFrame) -> None:
    total_col = _find_column(df, "Total_scaled")
    num_col = _numeric_column(df)
    total = df[total_col]
    numeric = df[num_col]
    ratio = numeric / total.replace(0, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.scatter(total, numeric, alpha=0.7)
    lims = [0, max(total.max(), numeric.max()) * 1.05]
    ax.plot(lims, lims, "k--")
    ax.set_xlabel("Dans intensity")
    ax.set_ylabel("Numeric intensity")
    ax.set_title("Numeric vs Dans")
    ax.set_aspect("equal", "box")
    ax.grid(True, ls=":", alpha=.4)

    axes[1].hist(ratio.dropna(), bins=20, edgecolor="k")
    axes[1].set_xlabel("Numeric / Dans")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Ratio distribution")
    axes[1].grid(axis="y", ls=":", alpha=.4)

    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Dans intensities with numeric Hendricks–Teller areas")
    parser.add_argument("excel_path", help="Path to miller_intensities.xlsx")
    parser.add_argument("--sheet", default="Summary", help="Worksheet name")
    args = parser.parse_args()

    df = pd.read_excel(Path(args.excel_path), sheet_name=args.sheet)
    metrics = compute_metrics(df)
    print(f"RMSE: {metrics['rmse']:.3f}")
    print(f"Mean ratio: {metrics['mean_ratio']:.3f}")
    plot_comparison(df)


if __name__ == "__main__":
    main()
