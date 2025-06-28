import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    from ra_sim.path_config import get_dir
except Exception:  # pragma: no cover
    get_dir = None


def _find_column(df: pd.DataFrame, target: str) -> str | None:
    """Return the name of ``target`` column in ``df`` if present.

    The search is case-insensitive and ignores spaces.  If an exact match is
    not found, a column containing the target text is returned if available.
    """

    normalized = target.lower().replace(" ", "")
    for col in df.columns:
        col_normalized = col.lower().replace(" ", "")
        if col_normalized == normalized:
            return col

    for col in df.columns:
        if normalized in col.lower().replace(" ", ""):
            return col
    return None


def _find_intensity_columns(df: pd.DataFrame, name: str | None) -> list[str]:
    """Return intensity columns based on ``name`` or heuristics.

    If ``name`` is ``None`` and multiple candidate columns are found,
    they are all returned instead of raising an error.
    """

    if name:
        col = _find_column(df, name)
        if not col:
            raise SystemExit(
                f"Intensity column '{name}' not found. Available columns: {list(df.columns)}"
            )
        return [col]

    col = _find_column(df, "Intensity")
    if col:
        return [col]

    keywords = ["scaled", "intensity", "area"]
    hkl_cols = {_find_column(df, "h"), _find_column(df, "k"), _find_column(df, "l")}
    candidates = [
        c
        for c in df.columns
        if any(k in c.lower() for k in keywords)
        and c not in hkl_cols
    ]

    if not candidates:
        raise SystemExit(
            f"Required column 'Intensity' not found. Available columns: {list(df.columns)}"
        )

    if len(candidates) > 1:
        print(
            "Multiple possible intensity columns found: "
            f"{candidates}. Plotting them all"
        )
        return candidates

    col = candidates[0]
    print(f"Using column '{col}' for intensities")
    return [col]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Miller intensities from an Excel file as a 3D scatter plot"
    )
    parser.add_argument(
        "excel_path",
        nargs="?",
        default=None,
        help="Path to miller_intensities.xlsx (defaults to downloads directory)",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="Name of worksheet to read (defaults to 'Summary' or first sheet)",
    )
    parser.add_argument(
        "--intensity",
        default=None,
        help="Column containing intensities (auto-detected if omitted)",
    )

    args = parser.parse_args()

    if args.excel_path is None:
        if get_dir is None:
            raise SystemExit("Excel path required when ra_sim is not installed")
        excel_path = Path(get_dir("downloads")) / "miller_intensities.xlsx"
    else:
        excel_path = Path(args.excel_path)

    if not excel_path.exists():
        raise SystemExit(f"File not found: {excel_path}")

    sheet_to_read = args.sheet or "Summary"
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_to_read)
    except ValueError as exc:
        xls = pd.ExcelFile(excel_path)
        available = xls.sheet_names
        if args.sheet is None and sheet_to_read == "Summary" and available:
            sheet_to_read = available[0]
            print(
                f"Worksheet 'Summary' not found; using '{sheet_to_read}' instead."
            )
            df = pd.read_excel(xls, sheet_name=sheet_to_read)
        else:
            raise SystemExit(
                f"Worksheet '{sheet_to_read}' not found. Available: {available}"
            ) from exc

    # Find required columns regardless of case or spaces
    required = ["h", "k", "l"]
    col_map = {}
    for col in required:
        found = _find_column(df, col)
        if not found:
            raise SystemExit(
                f"Required column '{col}' not found in sheet '{sheet_to_read}'. "
                f"Available columns: {list(df.columns)}"
            )
        col_map[col] = found

    intensity_cols = _find_intensity_columns(df, args.intensity)


    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = None
    for idx, col in enumerate(intensity_cols):
        sc = ax.scatter(
            df[col_map["h"]],
            df[col_map["k"]],
            df[col_map["l"]],
            c=df[col],
            cmap="viridis",
            alpha=0.7,
            label=col,
        )
        if idx == 0:
            fig.colorbar(sc, ax=ax, label="Normalized Intensity")
    ax.set_xlabel("h")
    ax.set_ylabel("k")
    ax.set_zlabel("l")
    if len(intensity_cols) > 1:
        ax.legend(title="Intensity columns")
    ax.set_title("Miller Intensities")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
