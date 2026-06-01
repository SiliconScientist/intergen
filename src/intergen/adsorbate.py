from ase import Atoms
from collections import Counter
import numpy as np
from pymatgen.core import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.analysis.structure_matcher import StructureMatcher
from intergen.config import Config
from intergen.surface import (
    prepare_for_pymatgen,
    get_atoms_per_layer,
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
    """Returns the indices of the adsorbate in the structure.

    The provided structure must already include the adsorbate appended to the slab.
    """
    structure_len = len(structure)
    adsorbate_len = len(adsorbate)
    adsorbate_indices = list(range(structure_len - adsorbate_len, structure_len))
    return adsorbate_indices


def get_host_symbol(structure: Structure, adsorbate_indices: list[int]) -> str:
    slab_symbols = [
        str(site.specie)
        for i, site in enumerate(structure)
        if i not in set(adsorbate_indices)
    ]
    return Counter(slab_symbols).most_common(1)[0][0]


def get_masked_adsorbate_substructure(
    structure: Structure,
    comparison_indices: list[int],
    adsorbate_indices: list[int],
    host_symbol: str,
) -> Structure:
    substructure = get_substructure(structure, indices=comparison_indices)
    adsorbate_index_set = set(adsorbate_indices)
    for local_index, parent_index in enumerate(comparison_indices):
        if parent_index not in adsorbate_index_set:
            substructure[local_index] = host_symbol
    return substructure


def normalize_fractional_coordinates(frac_coords: np.ndarray) -> np.ndarray:
    normalized = np.mod(frac_coords, 1.0)
    normalized[np.isclose(normalized, 1.0)] = 0.0
    return normalized


def get_adsorbate_canonical_key(
    structure: Structure, adsorbate_indices: list[int]
) -> tuple[float, ...]:
    adsorbate_coords = normalize_fractional_coordinates(
        np.array(structure.frac_coords[adsorbate_indices], dtype=float)
    )
    binding_xy = adsorbate_coords[0, :2]
    center_distance = float(np.sum((binding_xy - 0.5) ** 2))
    rounded_coords = tuple(np.round(adsorbate_coords.flatten(), 8))
    return (round(center_distance, 8),) + rounded_coords


def find_unique_adsorbate_structures(
    structures: list[Structure],
    surface_indices: list[int],
    adsorbate: Molecule,
    matcher: StructureMatcher,
) -> list[int]:
    if not structures:
        return []
    adsorbate_indices = get_adsorbate_indices(
        structure=structures[0], adsorbate=adsorbate
    )
    host_symbol = get_host_symbol(
        structure=structures[0], adsorbate_indices=adsorbate_indices
    )
    comparison_indices = surface_indices + adsorbate_indices[0:1]
    masked_substructures = [
        get_masked_adsorbate_substructure(
            structure=structure,
            comparison_indices=comparison_indices,
            adsorbate_indices=adsorbate_indices,
            host_symbol=host_symbol,
        )
        for structure in structures
    ]
    unique_indices = []
    unique_masked_substructures = []
    canonical_keys = []
    for index, (structure, masked_substructure) in enumerate(
        zip(structures, masked_substructures)
    ):
        candidate_key = get_adsorbate_canonical_key(
            structure=structure, adsorbate_indices=adsorbate_indices
        )
        duplicate_class = next(
            (
                class_index
                for class_index, unique_structure in enumerate(unique_masked_substructures)
                if matcher.fit(masked_substructure, unique_structure)
            ),
            None,
        )
        if duplicate_class is None:
            unique_indices.append(index)
            unique_masked_substructures.append(masked_substructure)
            canonical_keys.append(candidate_key)
            continue
        if candidate_key < canonical_keys[duplicate_class]:
            unique_indices[duplicate_class] = index
            unique_masked_substructures[duplicate_class] = masked_substructure
            canonical_keys[duplicate_class] = candidate_key
    return unique_indices


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
    atoms_per_layer = get_atoms_per_layer(cfg=cfg)
    slab_comparison_count = atoms_per_layer * cfg.adsorbate.surface_layers_for_matching
    for slab in slabs:
        structures = add_adsorbates(cfg=cfg, structure=slab, adsorbate=adsorbate)
        max_slab_index = min(len(slab), slab_comparison_count)
        surface_indices = list(range(max_slab_index))
        unique_indices = find_unique_adsorbate_structures(
            structures=structures,
            surface_indices=surface_indices,
            adsorbate=adsorbate,
            matcher=matcher,
        )
        unique_structures = [structures[i] for i in unique_indices]
        unique_atoms = [converter.get_atoms(struct) for struct in unique_structures]
        atoms_list.extend(unique_atoms)
    return atoms_list
