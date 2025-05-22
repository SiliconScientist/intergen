from ase.visualize import view
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Molecule
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.analysis.structure_matcher import StructureMatcher
from intergen.config import get_config
from intergen.surface import (
    build_pure_surfaces,
    naive_surface_index_selector,
    mutate_via_swaps,
)
from intergen.adsorbate import get_adsorbate_structures


def main():
    cfg = get_config()
    converter = AseAtomsAdaptor()
    matcher = StructureMatcher(**cfg.adsorbate.matcher.model_dump())
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
        matcher=matcher,
    )
    print(f"Generated {len(atoms_list)} unique structures.")
    adsorbate = Molecule(
        species=cfg.adsorbate.species,
        coords=cfg.adsorbate.coords,
        site_properties={
            "tags": [cfg.adsorbate.tag for _ in range(len(cfg.adsorbate.coords))]
        },
    )
    atoms_list = get_adsorbate_structures(
        cfg=cfg, atoms_list=atoms_list, adsorbate=adsorbate, matcher=matcher
    )
    print(f"Generated {len(atoms_list)} unique binding sites.")


if __name__ == "__main__":
    main()
