# Adsorption sites and matching

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

Validated use case (see the broader runtime check below):

- `reuse_site_templates_for_two_swap_motifs = true`
- `generation.num_swaps = 2`
- the slab is a `3x3` close-packed surface
- the top-layer motif is one of the validated classes: `heterodimer` or `dual_single_atom_alloy`

When the runtime check falls back:

- any other `num_swaps`
- a layer size other than nine atoms
- any other top-layer motif class
- when the config flag is disabled

Correctness assumption:

- motif-equivalent slabs in the validated cases share the same adsorption-site template, so a template discovered on one slab can be transferred cheaply to another slab of the same motif class.
- once a template exists for that motif class, the template becomes the canonical deduplicated site set for later slabs in the cache-hit path
- If that assumption does not hold for a case you care about, disable the fast path so every slab uses discovered sites followed by structure-matcher deduplication.

Tuning:

- `adsorbate.template_site_match_tolerance` controls how far a newly discovered raw site can be from a cached template site and still count as the same binding site.
- The default is `0.5`, which preserves the current regression-tested behavior. Tighten it if you want cache-hit mapping to be more conservative.
- The cache-hit matching policy favors speed over globally optimal assignment. If multiple nearby raw
  sites compete for multiple template sites, the nearest-first greedy choice wins.

## Implementation boundaries

The validated scope described above is narrower than the eligibility check: the code checks for nine atoms per layer, two swaps, and a supported motif, but does not explicitly check the surface family or that the in-plane dimensions are 3 × 3. Disable reuse for other geometries. Cache keys contain the host element and motif, not surface family, cell size, or adsorbate. The cache lasts for one `get_adsorbate_structures()` call. Avoid mixing incompatible geometries in that call.

The CLI path still discovers raw sites on every slab. Cache hits skip structure-matcher deduplication and select raw sites against the canonical template. Sites without a match within tolerance are omitted; this is not an automatic fallback to full deduplication. Timing output reports discovery, selection, placement, matching, and conversion separately.

Slab substitution deduplication compares only the top layer. Adsorbate deduplication compares the configured surface layers and only the first adsorbate atom. Molecular orientations and conformations are not exhaustively enumerated. The first adsorbate atom should be the intended binding atom.

An FCC or HCP hollow registry describes how a hollow lies relative to atoms beneath the surface. Keeping both registries distinct does not guarantee explicit `fcc_hollow` and `hcp_hollow` labels: the site finder may label both `hollow`.

[Documentation index](README.md)
