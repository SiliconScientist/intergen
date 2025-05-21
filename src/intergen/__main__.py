from ase.visualize import view

from intergen.config import get_config
from intergen.surface import (
    build_pure_surfaces,
    naive_surface_index_selector,
    mutate_via_swaps,
)


def main():
    cfg = get_config()
    pure_atoms = build_pure_surfaces(cfg=cfg)

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
    print("Number of surfaces:", len(atoms_list))


if __name__ == "__main__":
    main()
