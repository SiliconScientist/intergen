# Bulk constraints

`database.constrain_bottom_layers` controls whether generated adsorbate slabs are written with ASE `FixAtoms` constraints.

- Set `database.constrain_bottom_layers = 2` to freeze the bottom two slab layers in the written ASE DB.
- If `database.constrain_bottom_layers = 0`, `python -m intergen` will print a warning and write no `FixAtoms` constraints.
- Layer membership is determined geometrically from a best-fit plane through the lowest slab atoms, not from atom ordering alone.
- `database.constraint_z_tolerance` controls how heights are clustered into layers.
- `database.constraint_lowest_z_tolerance` controls which atoms define the bottom-plane fit.


These constraints prepare structures for later calculations; Intergen does not perform relaxation. Disabling constraint generation does not remove constraints already present in a reused database.

[Documentation index](README.md)
