# intergen
Flexible interface generator for catalysis, surface science, and materials modeling.

## Structure metadata

Generated slabs and adslabs carry explicit metadata in both `Atoms.info` and the written ASE DB rows.

Persisted slab provenance fields:
- `slab_id`: unique slab identifier assigned during surface generation
- `host_element`: majority top-layer host species for the slab family
- `surface_type`: normalized surface label, currently `fcc111` or `hcp0001`
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

## Bulk constraints

`database.constrain_bottom_layers` controls whether generated adsorbate slabs are written with ASE `FixAtoms` constraints.

- Set `database.constrain_bottom_layers = 2` to freeze the bottom two slab layers in the written ASE DB.
- If `database.constrain_bottom_layers = 0`, `python -m intergen` will print a warning and write no `FixAtoms` constraints.
- Layer membership is determined geometrically from a best-fit plane through the lowest slab atoms, not from atom ordering alone.
- `database.constraint_z_tolerance` controls how heights are clustered into layers.
- `database.constraint_lowest_z_tolerance` controls which atoms define the bottom-plane fit.

## Adsorbate matching

`adsorbate.surface_layers_for_matching` controls how many slab layers are used when deduplicating adsorbate structures.

Set `surface_layers_for_matching = 2` to distinguish FCC and HCP hollow sites on close-packed surfaces such as FCC(111). If you set it to `1`, those hollow registries collapse into a single unique site.

## Adsorption site template reuse

`adsorbate.reuse_site_templates_for_two_swap_motifs` enables a fast path for adsorption-site generation on validated two-swap surfaces.

What is cached:
- A motif-class adsorption-site template, not final adsorbate structures.
- Each cached site is stored as fractional in-plane coordinates plus height above the top-layer plane.
- On cache hits, raw discovered sites are mapped onto that template and the closest raw site within
  `adsorbate.template_site_match_tolerance` is kept for each template site.
- That mapping is intentionally greedy: valid template/raw pairs are sorted by distance, then accepted
  nearest-first without backtracking.

When the fast path is used:
- `reuse_site_templates_for_two_swap_motifs = true`
- `generation.num_swaps = 2`
- the slab is a `3x3` close-packed surface
- the top-layer motif is one of the validated classes: `heterodimer` or `dual_single_atom_alloy`

When it falls back:
- any other `num_swaps`
- any other surface size
- any other top-layer motif class
- when the config flag is disabled

Correctness assumption:
- motif-equivalent slabs in the validated cases share the same adsorption-site template, so a template discovered on one slab can be transferred cheaply to another slab of the same motif class.
- once a template exists for that motif class, the template becomes the canonical deduplicated site set for later slabs in the cache-hit path
- If that assumption does not hold for a case you care about, disable the fast path and the code will use direct `find_adsorption_sites()` calls for every slab.

Tuning:
- `adsorbate.template_site_match_tolerance` controls how far a newly discovered raw site can be from a cached template site and still count as the same binding site.
- The default is `0.5`, which preserves the current regression-tested behavior. Tighten it if you want cache-hit mapping to be more conservative.
- The cache-hit matching policy favors speed over globally optimal assignment. If multiple nearby raw
  sites compete for multiple template sites, the nearest-first greedy choice wins.
