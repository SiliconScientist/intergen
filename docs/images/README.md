# README structure visualization

The figure shows an actual Intergen-generated, unrelaxed Pt(111) slab with neighboring Cu and Ag substitutions and one N atom initially placed atop Cu. It is a geometry example, not a claim of stable adsorption or a structure identified as optimal in the publication.

[pt-cu-ag-n.extxyz](pt-cu-ag-n.extxyz) preserves the atomic coordinates, cell, element identities, fixed-atom constraints, and provenance from `adslab-000001` (parent `slab-000003`) of the documentation verification run. That run used the settings in [the trimetallic example](../examples/trimetallic.toml), with its output database named `data/quickstart.db`. The structure contains 34 Pt atoms, one Cu atom, one Ag atom, and one N atom in a 3 × 3, four-layer slab. The bottom two layers (18 Pt atoms) are fixed by Intergen using ASE `FixAtoms`, with `database.constrain_bottom_layers = 2`. The export preserves these constraints through the extended XYZ `move_mask` field.

Hatching identifies fixed atoms directly from the saved `FixAtoms` constraints; the upper two layers and N are unconstrained. The two views use the same coordinates and element colors: Pt gray, Cu orange, Ag purple, and N blue. The figure shows one cell’s atoms; periodic copies and the vacuum cell boundary are omitted for clarity. Display radii are chosen for legibility and do not represent measured atomic sizes. No bonds are inferred or drawn.

Regenerate [the PNG](pt-cu-ag-n.png) from the repository root with the installed ASE and Matplotlib dependencies:

```sh
python docs/images/render_structure.py
```

The [rendering script](render_structure.py) only reads the saved structure and draws it; it does not alter Intergen’s scientific implementation.

[Documentation index](../README.md)
