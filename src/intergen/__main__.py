from ase.visualize import view
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Molecule
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from intergen.config import get_config
from intergen.surface import (
    build_pure_surfaces,
    naive_surface_index_selector,
    mutate_via_swaps,
)


def main():
    cfg = get_config()
    converter = AseAtomsAdaptor()
    pure_atoms = build_pure_surfaces(cfg=cfg)
    slab = converter.get_structure(pure_atoms[0])
    site_finder = AdsorbateSiteFinder(slab=slab)
    site_coordinates = site_finder.find_adsorption_sites(symm_reduce=True)
    adsorbate = Molecule(
        species="OH",
        coords=[(0, 0, 0), (0, 0, 0.98)],
        site_properties={"tags": [0, 0]},
    )
    structure = site_finder.add_adsorbate(
        molecule=adsorbate, ads_coord=site_coordinates["ontop"][0]
    )
    atoms = converter.get_atoms(structure)
    print(f"Found {len(site_coordinates)} adsorption sites")
    # # TODO: Inject index selector function to support alternative selection strategies
    # index_selector_fn = naive_surface_index_selector
    # swap_indices = index_selector_fn(cfg=cfg)

    # # TODO: Inject surface generation method to support multiple generation strategies
    # surface_generator = mutate_via_swaps
    # atoms_list = surface_generator(
    #     cfg=cfg,
    #     swap_indices=swap_indices,
    #     pure_atoms=pure_atoms,
    # )


if __name__ == "__main__":
    main()
