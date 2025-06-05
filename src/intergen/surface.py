import copy
import numpy as np
import pandas as pd
from ase.db import connect
from ase.atoms import Atoms
from ase.build import fcc111, hcp0001
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher

from intergen.config import Config


def get_atoms_per_layer(cfg: Config) -> int:
    x, y, _ = cfg.structure.size
    atoms_per_layer = x * y
    return atoms_per_layer


def naive_surface_index_selector(cfg):
    """
    Assumes that atom indices are ordered from top to bottom in the structure,
    and selects atoms from the top layers specified by cfg.generation.layers_to_swap.
    """
    atoms_per_layer = get_atoms_per_layer(cfg)
    num_atoms = atoms_per_layer * cfg.generation.layers_to_swap
    return list(range(num_atoms))


def build_pure_surfaces(cfg: Config) -> list[Atoms]:
    pure_atoms = []
    lattice_constant_df = pd.read_csv(
        "assets/pure_metal_lattice_constants.csv",
        index_col=0,
        skiprows=1,
    )
    for host in cfg.structure.fcc_list:
        slab = fcc111(
            host,
            size=cfg.structure.size,
            vacuum=cfg.structure.vacuum,
            a=lattice_constant_df.loc[host, "FCC_LatticeConstant_PW91_1"],
        )[::-1]
        pure_atoms.append(slab)

    for host in cfg.structure.hcp_list:
        slab = hcp0001(
            host,
            size=cfg.structure.size,
            vacuum=cfg.structure.vacuum,
            a=lattice_constant_df.loc[host, "HCP_LatticeConstant_PW91_1"],
            c=lattice_constant_df.loc[host, "HCP_LatticeConstant_PW91_1"]
            * lattice_constant_df.loc[host, "HCP_c_over_a_PW91_1"],
        )[::-1]
        pure_atoms.append(slab)
    return pure_atoms


def id_unique_sites(
    atoms: Atoms,
    indices: list[int],
    masked_element: str,
    symprec: float = 0.05,
    angle_tolerance: float = 1.0,
    converter: AseAtomsAdaptor = AseAtomsAdaptor(),
) -> list[int]:
    structure = converter.get_structure(atoms)
    symmetry_object = SpacegroupAnalyzer(
        structure, symprec=symprec, angle_tolerance=angle_tolerance
    )
    masked_sites = np.where(np.array(atoms.get_chemical_symbols()) == masked_element)[0]
    equivalent_atoms = symmetry_object.get_symmetry_dataset()["equivalent_atoms"]
    unique_indices = np.unique([equivalent_atoms[i] for i in indices])
    unique_unmasked_indices = np.setdiff1d(unique_indices, masked_sites)
    return unique_unmasked_indices


def swap_atoms(atoms: Atoms, index: int, element: str) -> Atoms:
    new_atoms = copy.deepcopy(atoms)
    new_symbols = atoms.get_chemical_symbols()
    new_symbols[index] = element
    new_atoms.set_chemical_symbols(new_symbols)
    return new_atoms


def enumerate_unique_swaps(
    atoms: Atoms,
    host_element: str,
    indices: list[int],
    element: str,
) -> list[Atoms]:
    """
    Generate a list of structures with the given element substituted at each
    symmetry-unique site from the provided indices.

    Parameters:
        atoms (Atoms): Original atomic structure.
        indices (list[int]): Candidate site indices for substitution.
        element (str): Element symbol to substitute in.

    Returns:
        list[Atoms]: List of new structures with one substitution each.
    """
    unique_site_indices = id_unique_sites(
        atoms=atoms,
        indices=indices,
        masked_element=element,
    )
    atoms_list = []
    symbols = atoms.get_chemical_symbols()
    for index in unique_site_indices:
        if symbols[index] == host_element:
            new_atoms = swap_atoms(atoms=atoms, index=index, element=element)
            atoms_list.append(new_atoms)
    return atoms_list


def get_substructure(structure: Structure, indices: list[int]) -> Structure:
    """Returns a substructure of the given structure based on the provided indices."""
    return Structure.from_sites([structure[i] for i in indices])


def prepare_for_pymatgen(
    atoms_list: list[Atoms], converter: AseAtomsAdaptor = AseAtomsAdaptor()
) -> list[Structure]:
    """Converts a list of ASE Atoms objects to PyMatGen Structure objects,
    setting periodic boundary conditions before conversion."""
    structures = []
    for atoms in atoms_list:
        atoms.set_pbc(True)
        structure = converter.get_structure(atoms)
        structures.append(structure)
    return structures


def find_unique_structures(
    structures: list[Structure],
    matcher: StructureMatcher = StructureMatcher(),
) -> list[int]:
    """Finds unique structures in a list of structures using a StructureMatcher."""
    unique_indices = []
    for i, candidate in enumerate(structures):
        is_duplicate = any(
            matcher.fit(candidate, structures[j]) for j in unique_indices
        )
        if not is_duplicate:
            unique_indices.append(i)
    return unique_indices


def get_unique_atoms(
    atoms_list: list[Atoms],
    comparison_indices: list[int],
    converter: AseAtomsAdaptor = AseAtomsAdaptor(),
    matcher: StructureMatcher = StructureMatcher(),
) -> list[Atoms]:
    structures = prepare_for_pymatgen(atoms_list)
    substructures = [
        get_substructure(structure, indices=comparison_indices)
        for structure in structures
    ]
    unique_indices = find_unique_structures(structures=substructures, matcher=matcher)
    unique_structures = [structures[i] for i in unique_indices]
    unique_atoms = [converter.get_atoms(struct) for struct in unique_structures]
    return unique_atoms


def get_element_swaps(
    atoms: Atoms,
    indices: list[int],
    element: str,
    atoms_per_layer: int,
    converter: AseAtomsAdaptor = AseAtomsAdaptor(),
    matcher: StructureMatcher = StructureMatcher(),
) -> list[Atoms]:
    """Returns symmetry-unique structures from a single round of element substitutions."""
    swapped_structs = enumerate_unique_swaps(
        atoms=atoms, indices=indices, element=element
    )
    structures = prepare_for_pymatgen(swapped_structs)
    comparison_indices = range(atoms_per_layer)
    substructures = [
        get_substructure(structure, indices=comparison_indices)
        for structure in structures
    ]
    unique_indices = find_unique_structures(structures=substructures, matcher=matcher)
    unique_structures = [structures[i] for i in unique_indices]
    unique_atoms = [converter.get_atoms(struct) for struct in unique_structures]
    return unique_atoms


def iterative_swaps(
    cfg: Config,
    atoms: list[Atoms],
    host_element: str,
    indices: list[int],
    atoms_per_layer: int,
    matcher: StructureMatcher = StructureMatcher(),
    only_last_generation: bool = False,
) -> list[Atoms]:
    all_atoms = []
    current_generation = [atoms]
    for i, element in enumerate(cfg.generation.swap_plan):
        last_generation = i == len(cfg.generation.swap_plan) - 1
        next_generation = []
        for atoms in current_generation:
            swaps = enumerate_unique_swaps(
                atoms=atoms,
                host_element=host_element,
                indices=indices,
                element=element,
            )
            unique_atoms = get_unique_atoms(
                atoms_list=swaps,
                comparison_indices=range(atoms_per_layer),
                matcher=matcher,
            )
            next_generation.extend(unique_atoms)
        if last_generation and only_last_generation:
            all_atoms = next_generation
        elif not only_last_generation:
            all_atoms.extend(next_generation)
        current_generation = next_generation
    unique_atoms = get_unique_atoms(
        atoms_list=all_atoms,
        comparison_indices=range(atoms_per_layer),
        matcher=matcher,
    )
    return unique_atoms


def mutate_via_swaps(
    cfg: Config,
    pure_atoms: list[Atoms],
    swap_indices: list[int],
    matcher: StructureMatcher = StructureMatcher(),
) -> list[Atoms]:
    atoms_list = []
    for atoms in pure_atoms:
        atoms_list.append(atoms)
        for element in cfg.generation.swap_elements:
            element_swaps = get_element_swaps(
                cfg=cfg,
                atoms=atoms,
                indices=swap_indices,
                element=element,
                num_swaps=cfg.generation.num_swaps,
                matcher=matcher,
            )
            for swapped_atoms in element_swaps:
                atoms_list.append(swapped_atoms)
    return atoms_list
