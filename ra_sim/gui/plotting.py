import matplotlib.pyplot as plt


def setup_figure():
    """Return a Matplotlib ``Figure`` and ``Axes`` for the GUI."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    return fig, ax
