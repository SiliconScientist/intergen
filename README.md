# intergen
Flexible interface generator for catalysis, surface science, and materials modeling.

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
