import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
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

plt.figure(figsize=(8, 8))
plt.imshow(image, cmap=turbo_white0, origin="upper")
plt.xlabel("x-pixel")
plt.ylabel("y-pixel")
plt.colorbar(label="Intensity")
plt.title("Simulated diffraction pattern")
plt.tight_layout()
plt.show()
