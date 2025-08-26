import argparse
from pathlib import Path

import os
import pandas as pd
import matplotlib.pyplot as plt

if (
    plt.get_backend().lower().endswith("agg")
    and not os.environ.get("PYTEST_CURRENT_TEST")
):
    try:  # pragma: no cover - UI safeguard
        plt.switch_backend("TkAgg")
    except Exception:
        pass

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

    keywords = ["scaled", "intensity", "area"]
    hkl_cols = {_find_column(df, "h"), _find_column(df, "k"), _find_column(df, "l")}
    candidates = [
        c
        for c in df.columns
        if any(k in c.lower() for k in keywords)
        and c not in hkl_cols
    ]

    if not candidates:
        # Fallback: use any numeric columns excluding Miller indices
        numeric_candidates = [
            c
            for c in df.select_dtypes(include="number").columns
            if c not in hkl_cols
        ]
        if numeric_candidates:
            print(
                "No standard intensity column found. "
                f"Using numeric columns {numeric_candidates}"
            )
            candidates = numeric_candidates
        else:
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


def _normalize_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return a copy of ``df`` with ``cols`` scaled so each has a 0–100 range."""
    out = df.copy()
    for c in cols:
        max_val = out[c].max()
        if not pd.api.types.is_number(max_val) or max_val == 0:
            max_val = 1.0
        out[c] = 100.0 * out[c] / max_val
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot intensities from an Excel file as an L vs intensity scatter plot "
            "with interactive controls"
        )

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

    # Locate Miller index columns
    col_map = {}
    for col in ["h", "k", "l"]:
        found = _find_column(df, col)
        if col == "l" and not found:
            raise SystemExit(
                f"Required column '{col}' not found in sheet '{sheet_to_read}'. "
                f"Available columns: {list(df.columns)}"
            )
        col_map[col] = found

    intensity_cols = _find_intensity_columns(df, args.intensity)

    df = _normalize_columns(df, intensity_cols)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatters = []
    for col in intensity_cols:
        sc = ax.scatter(
            df[col_map["l"]],
            df[col],
            label=col,
            s=20,
            alpha=0.7,
        )
        scatters.append(sc)

    ax.set_ylabel("Normalized Intensity (0-100)")
    ax.set_ylim(0, 110)
    ax.set_xlabel("l")
    ax.set_title("L vs Intensity")

    legend = ax.legend(title="Intensity columns") if len(intensity_cols) > 1 else None

    annot_ax = ax

    # annotate HKL labels at each L position (once per unique L)
    if col_map.get("h") is not None and col_map.get("k") is not None:
        seen = set()
        for _, row in df.iterrows():
            l_val = row[col_map["l"]]
            if l_val in seen:
                continue
            seen.add(l_val)
            label = f"({int(row[col_map['h']])},{int(row[col_map['k']])},{int(l_val)})"
            annot_ax.annotate(
                label,
                (l_val, 102),
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    if legend:
        from matplotlib.widgets import CheckButtons, Button

        rax = fig.add_axes([0.82, 0.4, 0.15, 0.2])
        visibility = [sc.get_visible() for sc in scatters]
        checks = CheckButtons(rax, intensity_cols, visibility)

        def func(label: str) -> None:
            idx = intensity_cols.index(label)
            scatters[idx].set_visible(not scatters[idx].get_visible())
            fig.canvas.draw_idle()

        checks.on_clicked(func)

        hide_ax = fig.add_axes([0.82, 0.32, 0.07, 0.05])
        show_ax = fig.add_axes([0.90, 0.32, 0.07, 0.05])
        hide_btn = Button(hide_ax, "Hide\nAll")
        show_btn = Button(show_ax, "Show\nAll")

        def hide_all(event) -> None:  # pragma: no cover - UI
            for i, sc in enumerate(scatters):
                if sc.get_visible():
                    checks.set_active(i)

        def show_all(event) -> None:  # pragma: no cover - UI
            for i, sc in enumerate(scatters):
                if not sc.get_visible():
                    checks.set_active(i)

        hide_btn.on_clicked(hide_all)
        show_btn.on_clicked(show_all)

    # Leave room on the right so widget axes remain clickable
    plt.subplots_adjust(right=0.78)
    plt.show()


if __name__ == "__main__":
    main()
