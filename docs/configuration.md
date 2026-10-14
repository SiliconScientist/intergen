# Configuration and scientific scope

[Documentation index](README.md)

Use [the small, complete example](examples/quickstart.toml) as a starting point. `python -m intergen` reads `config.toml` in the current working directory; there is no command-line config-path option. That directory must also contain `assets/pure_metal_lattice_constants.csv`. Create the output directory before running.

The older root [config.example.toml](../config.example.toml) preserves a larger substitution list, but needs a `[structure]` header before `vacuum` to parse with the current schema. Its `adsorbate.sites` entry is accepted but does not filter site discovery. Prefer the example under `docs/` for new runs.

## Surface models

| Setting | Meaning |
| --- | --- |
| `structure.vacuum` | Vacuum thickness in Å on each side of the slab, passed to ASE. |
| `structure.fcc.elements` | Host elements for FCC(111). |
| `structure.bcc.elements` | Host elements for BCC(111). |
| `structure.hcp.elements` | Host elements for HCP(0001). |
| `structure.<family>.size` | In-plane repetitions and layer count, e.g. `[3, 3, 4]`. Must satisfy the corresponding ASE builder’s requirements. |

All three family tables are required; an empty `elements` list disables a family. Host elements must have suitable numeric entries in the bundled lattice-constant CSV. FCC and BCC use its PBE+TS lattice constants; HCP uses its PW91 lattice constant and c/a ratio. These are starting geometries, not a consistent recalculation of all hosts under one electronic-structure method.

## Alloy generation

| Setting | Meaning |
| --- | --- |
| `generation.layers_to_swap` | Number of top layers eligible for substitution. Use `1` for the documented surface-alloy workflow. |
| `generation.num_swaps` | Number of successive substitutions. Use `0` for pure surfaces, `1` for single substitutions, or `2` for trimetallic candidates. |
| `generation.swap_elements` | Candidate substituents. Plans are combinations without replacement; plans containing the host are excluded. Supply distinct element symbols. |
| `generation.only_last_generation` | `true` retains only the final substitution stage; `false` also includes pure slabs and intermediate stages before adsorbate placement. |

The [trimetallic example](examples/trimetallic.toml) expands the quickstart to two substitutions and a separate output database. For example, host Pt, `swap_elements = ["Cu", "Ag"]`, and `num_swaps = 2` produce one Cu and one Ag substitution per generated slab. This does not enumerate arbitrary alloy compositions or repeated substitutions of the same element. With too few eligible substituents, no final structures may be generated. Deduplication is performed within each swap plan, not globally across all plans.

## Adsorbates and matching

| Setting | Meaning |
| --- | --- |
| `adsorbate.species` | Species string passed to pymatgen `Molecule`, e.g. `"N"` or `"OH"`. |
| `adsorbate.coords` | One Cartesian coordinate in Å per adsorbate atom. |
| `adsorbate.tag` | Integer tag supplied as a molecule site property. |
| `adsorbate.surface_layers_for_matching` | Number of top slab layers included in adsorption-structure matching; use `2` to distinguish FCC/HCP hollow registries. |
| `adsorbate.sites` | Optional legacy field; not used to select sites in the current generation path. |
| `adsorbate.matcher.ltol` | Fractional lattice-length tolerance for pymatgen `StructureMatcher`. |
| `adsorbate.matcher.stol` | Site tolerance normalized by the average free length per atom. |
| `adsorbate.matcher.angle_tol` | Lattice-angle tolerance in degrees. |
| `adsorbate.reuse_site_templates_for_two_swap_motifs` | Enables the conditional template path; defaults to `true`. The quickstart disables it. |
| `adsorbate.template_site_match_tolerance` | Cartesian template matching tolerance in Å; defaults to `0.5`. |

For OH, replace the example’s species and coordinates together:

```toml
species = "OH"
coords = [[0.0, 0.0, 0.0], [0.937, 0.0, 0.287]]
```

These lines belong inside `[adsorbate]`. O is the first atom and intended binding atom. The coordinates supply one starting molecular geometry, not an orientation search. Site placement uses pymatgen’s defaults; placement height and orientation sampling are not exposed as configuration options. See [site generation](site-generation.md) for matching and template assumptions.

## Database and reruns

`database.path` selects the ASE database. The CLI writes adsorbate–slab structures, not a separate database of bare parent slabs. Structure metadata is described in [the metadata reference](metadata.md).

`database.constrain_bottom_layers` defaults to `0`; set it to `2` to freeze the bottom two layers. Both `database.constraint_z_tolerance` and `database.constraint_lowest_z_tolerance` default to `0.5` Å. See [bulk constraints](constraints.md).

If a nonempty database already exists, the CLI reads it instead of regenerating structures. It skips writing if the requested constraints are already present; otherwise it rewrites the database with the loaded structures and added constraints. This is not a check that the stored structures match the current configuration. Use a fresh output path after changing generation settings, and preserve valuable databases before constraint updates. Setting the constraint count to zero does not remove stored constraints.

## Scientific scope and limitations

- Enumeration is bounded by the selected hosts, surface orientations, finite periodic cells, substitution plans, and site-finder output. It is not proof of completeness over all trimetallic materials or binding geometries.
- Motif classification supports at most two non-host atoms in the top layer. Neighboring substituted atoms are labeled `heterodimer`; separated ones are labeled `dual_single_atom_alloy`. Those labels describe geometry, not stability.
- Substitution changes element identities while retaining the host lattice and initial positions. No alloy lattice optimization, surface reconstruction, segregation calculation, or thermodynamic stability test is performed.
- Slab deduplication compares the top layer even when deeper substitutions are requested. Deeper-layer distinctions can therefore be lost; use top-layer substitutions for the documented workflow.
- Adsorbate matching includes only the first adsorbate atom and the selected surface layers. With one layer, FCC/HCP hollow registries collapse. Molecular conformations, orientations, multiple-adsorbate coverages, and all possible local minima are not enumerated.
- Template reuse assumes equivalent motifs share a canonical site set. Its runtime eligibility check is broader than the validated close-packed cases; consult [the implementation boundaries](site-generation.md#implementation-boundaries) before enabling it elsewhere.
- Output contains initial structures and constraints, not energies, forces, relaxed geometries, or catalytic activity predictions. Those require external calculations and validation.
