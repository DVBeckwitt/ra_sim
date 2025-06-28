import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    from ra_sim.path_config import get_dir
except Exception:  # pragma: no cover
    get_dir = None


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

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(df["h"], df["k"], df["l"], c=df["Intensity"], cmap="viridis")
    ax.set_xlabel("h")
    ax.set_ylabel("k")
    ax.set_zlabel("l")
    fig.colorbar(sc, ax=ax, label="Normalized Intensity")
    ax.set_title("Miller Intensities")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
