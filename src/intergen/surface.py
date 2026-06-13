from collections.abc import Iterator, Sequence
from itertools import combinations, count
from typing import Literal

import numpy as np
import pandas as pd
from ase.atoms import Atoms
from ase.build import fcc111, hcp0001
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from intergen.config import Config
from intergen.metadata import (
    HOST_ELEMENT_KEY,
    SLAB_ID_KEY,
    SUPERCELL_SIZE_KEY,
    SURFACE_TYPE_FCC111,
    SURFACE_TYPE_HCP0001,
    SURFACE_TYPE_KEY,
    SWAP_ELEMENTS_KEY,
    SWAP_INDICES_KEY,
    TOP_LAYER_MOTIF_KEY,
    TOP_LAYER_MOTIF_PURE,
    validate_structure_metadata_keys,
)


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


def make_slab_id(slab_id_source: Iterator[int]) -> str:
    return f"slab-{next(slab_id_source):06d}"


def build_slab_metadata(
    slab_id: str,
    host_element: str,
    surface_type: str,
    supercell_size: Sequence[int],
    swap_indices: Sequence[int] = (),
    swap_elements: Sequence[str] = (),
) -> dict[str, object]:
    metadata = {
        SLAB_ID_KEY: slab_id,
        HOST_ELEMENT_KEY: host_element,
        SURFACE_TYPE_KEY: surface_type,
        SUPERCELL_SIZE_KEY: tuple(int(value) for value in supercell_size),
        SWAP_INDICES_KEY: [int(index) for index in swap_indices],
        SWAP_ELEMENTS_KEY: [str(element) for element in swap_elements],
    }
    validate_structure_metadata_keys(metadata)
    return metadata


def get_metadata_atoms_per_layer(atoms: Atoms) -> int:
    x_size, y_size, _ = atoms.info[SUPERCELL_SIZE_KEY]
    return x_size * y_size


def assign_top_layer_motif_metadata(atoms: Atoms) -> Atoms:
    atoms.info[TOP_LAYER_MOTIF_KEY] = classify_top_layer_motif(
        atoms=atoms,
        atoms_per_layer=get_metadata_atoms_per_layer(atoms),
    )
    return atoms


def assign_slab_metadata(
    atoms: Atoms,
    *,
    slab_id: str,
    host_element: str,
    surface_type: str,
    supercell_size: Sequence[int],
    swap_indices: Sequence[int] = (),
    swap_elements: Sequence[str] = (),
    top_layer_motif: str | None = None,
) -> Atoms:
    metadata = build_slab_metadata(
        slab_id=slab_id,
        host_element=host_element,
        surface_type=surface_type,
        supercell_size=supercell_size,
        swap_indices=swap_indices,
        swap_elements=swap_elements,
    )
    if top_layer_motif is not None:
        metadata[TOP_LAYER_MOTIF_KEY] = top_layer_motif
        validate_structure_metadata_keys(metadata)
    atoms.info.update(metadata)
    return atoms


def derive_swapped_slab_metadata(
    parent_atoms: Atoms,
    *,
    slab_id: str,
    swap_index: int,
    swap_element: str,
) -> dict[str, object]:
    return build_slab_metadata(
        slab_id=slab_id,
        host_element=parent_atoms.info[HOST_ELEMENT_KEY],
        surface_type=parent_atoms.info[SURFACE_TYPE_KEY],
        supercell_size=parent_atoms.info[SUPERCELL_SIZE_KEY],
        swap_indices=[*parent_atoms.info[SWAP_INDICES_KEY], int(swap_index)],
        swap_elements=[*parent_atoms.info[SWAP_ELEMENTS_KEY], str(swap_element)],
    )


def build_pure_surfaces(
    cfg: Config,
    slab_id_source: Iterator[int] | None = None,
) -> list[Atoms]:
    if slab_id_source is None:
        slab_id_source = count(1)
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
            a=lattice_constant_df.loc[host, "FCC_LatticeConstant_PBE+TS_1"],
        )[::-1]
        assign_slab_metadata(
            slab,
            slab_id=make_slab_id(slab_id_source),
            host_element=host,
            surface_type=SURFACE_TYPE_FCC111,
            supercell_size=cfg.structure.size,
            top_layer_motif=TOP_LAYER_MOTIF_PURE,
        )
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
        assign_slab_metadata(
            slab,
            slab_id=make_slab_id(slab_id_source),
            host_element=host,
            surface_type=SURFACE_TYPE_HCP0001,
            supercell_size=cfg.structure.size,
            top_layer_motif=TOP_LAYER_MOTIF_PURE,
        )
        pure_atoms.append(slab)
    return pure_atoms


TopLayerMotif = Literal[
    "pure",
    "single_swap",
    "heterodimer",
    "dual_single_atom_alloy",
]


def classify_top_layer_motif(atoms: Atoms, atoms_per_layer: int) -> TopLayerMotif:
    top_layer_symbols = atoms.get_chemical_symbols()[:atoms_per_layer]
    host_element = max(set(top_layer_symbols), key=top_layer_symbols.count)
    swapped_indices = [
        index
        for index, symbol in enumerate(top_layer_symbols)
        if symbol != host_element
    ]
    num_swaps = len(swapped_indices)
    if num_swaps == 0:
        return "pure"
    if num_swaps == 1:
        return "single_swap"
    if num_swaps != 2:
        raise ValueError(
            "Top-layer motif classification only supports up to two swaps."
        )

    top_layer = atoms[:atoms_per_layer]
    swapped_distance = top_layer.get_distance(
        swapped_indices[0], swapped_indices[1], mic=True
    )
    pair_distances = []
    for i in range(atoms_per_layer):
        for j in range(i + 1, atoms_per_layer):
            distance = top_layer.get_distance(i, j, mic=True)
            if not np.isclose(distance, 0.0):
                pair_distances.append(distance)
    nearest_neighbor_distance = min(pair_distances)
    if np.isclose(swapped_distance, nearest_neighbor_distance):
        return "heterodimer"
    return "dual_single_atom_alloy"


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
    equivalent_atoms = symmetry_object.get_symmetry_dataset().equivalent_atoms
    unique_indices = np.unique([equivalent_atoms[i] for i in indices])
    unique_unmasked_indices = np.setdiff1d(unique_indices, masked_sites)
    return unique_unmasked_indices


def swap_atoms(atoms: Atoms, index: int, element: str) -> Atoms:
    new_atoms = atoms.copy()
    new_symbols = new_atoms.get_chemical_symbols()
    new_symbols[index] = element
    new_atoms.set_chemical_symbols(new_symbols)
    return new_atoms


def enumerate_unique_swaps(
    atoms: Atoms,
    host_element: str,
    indices: list[int],
    element: str,
    slab_id_source: Iterator[int] | None = None,
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
    if slab_id_source is None:
        slab_id_source = count(1)
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
            new_atoms.info.update(
                derive_swapped_slab_metadata(
                    atoms,
                    slab_id=make_slab_id(slab_id_source),
                    swap_index=index,
                    swap_element=element,
                )
            )
            assign_top_layer_motif_metadata(new_atoms)
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
    return [atoms_list[i].copy() for i in unique_indices]


def get_element_swaps(
    atoms: Atoms,
    host_element: str,
    indices: list[int],
    element: str,
    atoms_per_layer: int,
    converter: AseAtomsAdaptor = AseAtomsAdaptor(),
    matcher: StructureMatcher = StructureMatcher(),
) -> list[Atoms]:
    """Returns symmetry-unique structures from a single round of element substitutions."""
    swapped_structs = enumerate_unique_swaps(
        atoms=atoms,
        host_element=host_element,
        indices=indices,
        element=element,
    )
    return get_unique_atoms(
        atoms_list=swapped_structs,
        comparison_indices=range(atoms_per_layer),
        converter=converter,
        matcher=matcher,
    )


def get_swap_plans(cfg: Config, num_swaps: int, host_element: str) -> list[list[str]]:
    swap_combinations = [
        list(pair)
        for pair in combinations(cfg.generation.swap_elements, num_swaps)
        if host_element not in pair
    ]
    return swap_combinations


def iterative_swaps(
    atoms: list[Atoms],
    host_element: str,
    swap_plan: list[str],
    indices: list[int],
    atoms_per_layer: int,
    slab_id_source: Iterator[int] | None = None,
    matcher: StructureMatcher = StructureMatcher(),
    only_last_generation: bool = False,
) -> list[Atoms]:
    if slab_id_source is None:
        slab_id_source = count(1)
    all_atoms = []
    current_generation = [atoms]
    comparison_indices = range(atoms_per_layer)
    for i, element in enumerate(swap_plan):
        last_generation = i == len(swap_plan) - 1
        raw_next_generation = []
        for current_atoms in current_generation:
            raw_next_generation.extend(
                enumerate_unique_swaps(
                    atoms=current_atoms,
                    host_element=host_element,
                    indices=indices,
                    element=element,
                    slab_id_source=slab_id_source,
                )
            )
        next_generation = get_unique_atoms(
            atoms_list=raw_next_generation,
            comparison_indices=comparison_indices,
            matcher=matcher,
        )
        if last_generation and only_last_generation:
            all_atoms = next_generation
        elif not only_last_generation:
            all_atoms.extend(next_generation)
        current_generation = next_generation
    unique_atoms = get_unique_atoms(
        atoms_list=all_atoms,
        comparison_indices=comparison_indices,
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
                atoms=atoms,
                host_element=atoms.info[HOST_ELEMENT_KEY],
                indices=swap_indices,
                element=element,
                atoms_per_layer=get_atoms_per_layer(cfg),
                matcher=matcher,
            )
            for swapped_atoms in element_swaps:
                atoms_list.append(swapped_atoms)
    return atoms_list
