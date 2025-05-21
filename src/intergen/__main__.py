from ase.visualize import view
from acat.adsorption_sites import SlabAdsorptionSites

from intergen.config import get_config
from intergen.surface import (
    build_pure_surfaces,
    naive_surface_index_selector,
    mutate_via_swaps,
)


def main():
    cfg = get_config()
    pure_atoms = build_pure_surfaces(cfg=cfg)
    sas = SlabAdsorptionSites(
        pure_atoms[0],
        surface="fcc111",
        allow_6fold=False,
        composition_effect=True,
        both_sides=False,
        label_sites=True,
        surrogate_metal="Cu",
    )
    # TODO: Inject index selector function to support alternative selection strategies
    index_selector_fn = naive_surface_index_selector
    swap_indices = index_selector_fn(cfg=cfg)

    # TODO: Inject surface generation method to support multiple generation strategies
    surface_generator = mutate_via_swaps
    atoms_list = surface_generator(
        cfg=cfg,
        swap_indices=swap_indices,
        pure_atoms=pure_atoms,
    )
    print(f"Generated {len(atoms_list)} surfaces")


if __name__ == "__main__":
    main()
