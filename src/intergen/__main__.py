from ase.visualize import view
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Molecule
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from intergen.config import get_config
from intergen.surface import (
    build_pure_surfaces,
    naive_surface_index_selector,
    mutate_via_swaps,
    get_atoms_per_layer,
    find_unique_structures,
    get_substructure,
    prepare_for_pymatgen,
)


def main():
    cfg = get_config()
    converter = AseAtomsAdaptor()
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
    print(f"Generated {len(atoms_list)} unique structures.")
    slabs = prepare_for_pymatgen(atoms_list)
    all_binding_sites = []
    for slab in slabs:
        site_finder = AdsorbateSiteFinder(slab=slab)
        site_coordinates = site_finder.find_adsorption_sites(symm_reduce=False)
        adsorbate = Molecule(
            species=cfg.adsorbate.species,
            coords=cfg.adsorbate.coords,
            site_properties={
                "tags": [cfg.adsorbate.tag for _ in range(len(cfg.adsorbate.coords))]
            },
        )
        structures = []
        for site in site_coordinates["ontop"]:
            structure = site_finder.add_adsorbate(molecule=adsorbate, ads_coord=site)
            structures.append(structure)
        adsorbate_indices = [i for i in range(len(structure))][-2:]
        surface_indices = list(range(get_atoms_per_layer(cfg=cfg)))
        comparison_indices = surface_indices + adsorbate_indices
        subsubstructures = [
            get_substructure(structure, indices=comparison_indices)
            for structure in structures
        ]
        unique_indices = find_unique_structures(structures=subsubstructures)
        unique_structures = [structures[i] for i in unique_indices]
        unique_atoms = [converter.get_atoms(struct) for struct in unique_structures]
        all_binding_sites.extend(unique_atoms)
    print(f"Generated {len(all_binding_sites)} unique binding sites.")


if __name__ == "__main__":
    main()
