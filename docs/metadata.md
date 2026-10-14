# Structure metadata

Generated slabs and adslabs carry explicit metadata in both `Atoms.info` and the written ASE DB rows.

Persisted slab provenance fields:

- `slab_id`: unique slab identifier assigned during surface generation
- `host_element`: majority top-layer host species for the slab family
- `surface_type`: normalized surface label, currently `fcc111`, `bcc111`, or `hcp0001`
- `supercell_size`: slab supercell as `(x, y, z)`
- `swap_indices`: top-layer substitution indices in generation order
- `swap_elements`: substituted elements in generation order
- `top_layer_motif`: `pure`, `single_swap`, `heterodimer`, or `dual_single_atom_alloy`

Persisted adslab fields:

- `adslab_id`: unique adsorbate-slab identifier assigned when final ASE structures are emitted
- `parent_slab_id`: slab provenance link back to the source slab
- `adsorbate`: adsorbate identity from the input molecule, such as `N` or `HO`
- `initial_site_label`: normalized initial adsorption-site label
- `initial_site_coordinate`: initial adsorption coordinate before any relaxation

Notes:

- `initial_site_label` reflects what the generator actually knows at placement time. Today that can include generic `hollow` in addition to `top`, `bridge`, `fcc_hollow`, and `hcp_hollow`.
- Adsorption-site placement follows whatever supported site classes `AdsorbateSiteFinder` actually returns for a slab. The config no longer needs a manual site list.
- Metadata is written twice on purpose:
  - scalar/query-friendly values are written as ASE DB row key-value pairs
  - the full typed metadata payload is also stored in row `data["structure_metadata"]` for clean round-trips of tuples and lists

Example ASE DB queries:

```python
from ase.db import connect

db = connect("data/intergen.db")

# All heterodimer-derived adslabs with N adsorbed.
rows = list(db.select(top_layer_motif="heterodimer", adsorbate="N"))

# All adslabs generated from one parent slab.
rows = list(db.select(parent_slab_id="slab-000123"))

# All structures initially placed at a specific site class.
rows = list(db.select(initial_site_label="fcc_hollow"))

# Inspect the full typed metadata payload for one row.
row = next(db.select(adslab_id="adslab-000001"))
metadata = row.data["structure_metadata"]
```


Identifiers are assigned within a generation run, not globally across databases. The CLI writes adsorbate–slab structures; parent slabs are identified by metadata but are not separate rows in that output. Use `get_database_atoms_list()` in `intergen.__main__` to restore metadata into `Atoms.info`; plain `row.toatoms()` does not perform that restoration.

The query values above are illustrative and need not exist in every dataset.

[Documentation index](README.md)
