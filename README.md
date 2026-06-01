# intergen
Flexible interface generator for catalysis, surface science, and materials modeling.

## Adsorbate matching

`adsorbate.surface_layers_for_matching` controls how many slab layers are used when deduplicating adsorbate structures.

Set `surface_layers_for_matching = 2` to distinguish FCC and HCP hollow sites on close-packed surfaces such as FCC(111). If you set it to `1`, those hollow registries collapse into a single unique site.
