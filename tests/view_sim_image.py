import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Slider
from pathlib import Path

NPZ_PATH = Path(__file__).resolve().parent / "simulation.npz"

if not NPZ_PATH.exists():
    raise SystemExit(f"File not found: {NPZ_PATH}")

with np.load(NPZ_PATH, allow_pickle=True) as data:
    if "image" not in data:
        raise SystemExit("image entry not found in npz file")
    image = data["image"]

# setup same colormap as run_diffraction_test.py
cmap = cm.get_cmap("turbo", 256)
rgba = cmap(np.linspace(0, 1, 256))
rgba[0] = [1, 1, 1, 1]
turbo_white0 = ListedColormap(rgba, name="turbo_white0")
turbo_white0.set_bad("white")

fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.15)
im = ax.imshow(image, cmap=turbo_white0, origin="upper")
ax.set_xlabel("x-pixel")
ax.set_ylabel("y-pixel")
cbar = plt.colorbar(im, ax=ax, label="Intensity")
ax.set_title("Simulated diffraction pattern")

# Slider for adjusting the maximum display intensity
slider_ax = fig.add_axes([0.2, 0.04, 0.6, 0.03])
vmax_init = float(np.max(image))
vmax_slider = Slider(slider_ax, "vmax", 0.0, vmax_init, valinit=vmax_init)

def on_vmax_change(val):
    im.set_clim(vmin=0.0, vmax=val)
    fig.canvas.draw_idle()

vmax_slider.on_changed(on_vmax_change)

plt.show()
