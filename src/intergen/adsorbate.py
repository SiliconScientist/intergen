from ase import Atoms
from pymatgen.core import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.analysis.structure_matcher import StructureMatcher
from intergen.config import Config
from intergen.surface import (
    prepare_for_pymatgen,
    get_atoms_per_layer,
    find_unique_structures,
    get_substructure,
)


def add_adsorbates(cfg: Config, structure, adsorbate: Molecule) -> list[Structure]:
    site_finder = AdsorbateSiteFinder(slab=structure)
    site_coordinates = site_finder.find_adsorption_sites(symm_reduce=False)
    structures = []
    for site in cfg.adsorbate.sites:
        for ads_coords in site_coordinates[site]:
            structure = site_finder.add_adsorbate(
                molecule=adsorbate, ads_coord=ads_coords
            )
            structures.append(structure)
    return structures


def get_adsorbate_indices(structure: Structure, adsorbate: Molecule) -> list[int]:
    """Returns the indices of the adsorbate in the structure."""
    structure_len = len(structure)
    adsorbate_len = len(adsorbate)
    adsorbate_indices = list(
        range(structure_len - 1, structure_len + adsorbate_len - 1)
    )
    return adsorbate_indices


def get_adsorbate_structures(
    cfg: Config,
    atoms_list: list[Atoms],
    adsorbate: Molecule,
    matcher: StructureMatcher,
    converter: AseAtomsAdaptor = AseAtomsAdaptor(),
):
    """Generates all unique structures with the specified adsorbate."""
    slabs = prepare_for_pymatgen(atoms_list)
    atoms_list = []
    for slab in slabs:
        structures = add_adsorbates(cfg=cfg, structure=slab, adsorbate=adsorbate)
        adsorbate_indices = get_adsorbate_indices(structure=slab, adsorbate=adsorbate)
        surface_indices = list(range(get_atoms_per_layer(cfg=cfg)))
        comparison_indices = (
            surface_indices + adsorbate_indices[0:1]
        )  # Why [0:1]? -> Only use binding atom index for structure matching
        subsubstructures = [
            get_substructure(structure, indices=comparison_indices)
            for structure in structures
        ]
        unique_indices = find_unique_structures(
            structures=subsubstructures, matcher=matcher
        )
        unique_structures = [structures[i] for i in unique_indices]
        unique_atoms = [converter.get_atoms(struct) for struct in unique_structures]
        atoms_list.extend(unique_atoms)
    return atoms_list
