"""Render the accompanying Intergen structure; run from the repository root."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ase.io import read
from ase.visualize.plot import Matplotlib
from ase.constraints import FixAtoms
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
atoms = read(HERE / "pt-cu-ag-n.extxyz")
colors = {"Pt": "#aebbc7", "Cu": "#d88436", "Ag": "#9970ab", "N": "#2075b8"}
radii = {"Pt": 1.10, "Cu": 1.00, "Ag": 1.10, "N": 0.65}
symbols = atoms.get_chemical_symbols()
fixed = {int(i) for constraint in atoms.constraints if isinstance(constraint, FixAtoms)
         for i in constraint.get_indices()}
fig, axes = plt.subplots(1, 2, figsize=(10, 5.1), facecolor="white")
for ax, rotation, title in zip(axes, ["0x,0y,0z", "-80x,0y,0z"], ["Top view", "Oblique view"]):
    drawing = Matplotlib(atoms, ax, rotation=rotation, show_unit_cell=0,
               colors=[colors[s] for s in symbols], radii=[radii[s] for s in symbols])
    drawing.write()
    # ASE draws fully occupied atoms in depth order; no cell edges are shown.
    order = drawing.positions[:len(atoms), 2].argsort()
    assert len(ax.patches) == len(atoms)
    for index, patch in zip(order, ax.patches):
        patch.set_linewidth(0.8)
        if int(index) in fixed:
            patch.set_hatch("///")
            patch.set_edgecolor("#536471")
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, color="#334155", pad=12)
fig.suptitle("Nitrogen on a Pt–Cu–Ag surface", fontsize=17, color="#172b40", y=0.98)
labels = {"Pt": "Pt · host", "Cu": "Cu · substitution", "Ag": "Ag · substitution", "N": "N · adsorbate"}
handles = [Line2D([], [], marker="o", linestyle="none", markersize=10,
                   markerfacecolor=colors[s], markeredgecolor="#334155", label=labels[s])
           for s in colors]
handles.append(Patch(facecolor=colors["Pt"], edgecolor="#536471", hatch="///",
                     label="Fixed · bottom two layers"))
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10,
           bbox_to_anchor=(0.5, 0.015))
fig.subplots_adjust(left=0.025, right=0.975, bottom=0.21, top=0.84, wspace=0.12)
fig.savefig(HERE / "pt-cu-ag-n.png", dpi=200, facecolor="white")
plt.close(fig)
