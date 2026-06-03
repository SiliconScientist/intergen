from dataclasses import dataclass
from time import perf_counter

import numpy as np
from ase import Atoms
from pymatgen.core import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.analysis.structure_matcher import StructureMatcher
from intergen.config import Config
from intergen.surface import (
    classify_top_layer_motif,
    prepare_for_pymatgen,
    get_atoms_per_layer,
    find_unique_structures,
    get_substructure,
)


@dataclass
class AdsorbateGenerationStats:
    slabs_processed: int = 0
    site_finder_calls: int = 0
    site_finding_seconds: float = 0.0
    matching_seconds: float = 0.0

    def summary(self) -> str:
        return (
            "Adsorbate generation stats: "
            f"slabs={self.slabs_processed}, "
            f"site_finder_calls={self.site_finder_calls}, "
            f"site_finding_seconds={self.site_finding_seconds:.3f}, "
            f"matching_seconds={self.matching_seconds:.3f}"
        )


MOTIF_SITE_CACHEABLE = {"heterodimer", "dual_single_atom_alloy"}


@dataclass
class AdsorptionSiteTemplate:
    coordinates_by_site: dict[str, list[tuple[float, float, float]]]


def supports_two_swap_motif_template_reuse(cfg: Config, motif: str) -> bool:
    return (
        cfg.adsorbate.reuse_site_templates_for_two_swap_motifs
        and cfg.generation.num_swaps == 2
        and cfg.structure.size[:2] == (3, 3)
        and motif in MOTIF_SITE_CACHEABLE
    )


def add_adsorbates(
    cfg: Config,
    structure,
    adsorbate: Molecule,
    stats: AdsorbateGenerationStats | None = None,
) -> list[Structure]:
    site_coordinates = discover_adsorption_sites(structure=structure, stats=stats)
    return apply_adsorption_sites(
        cfg=cfg,
        structure=structure,
        adsorbate=adsorbate,
        site_coordinates=site_coordinates,
    )


def discover_adsorption_sites(
    structure,
    stats: AdsorbateGenerationStats | None = None,
) -> dict[str, list]:
    site_finder = AdsorbateSiteFinder(slab=structure)
    start = perf_counter()
    site_coordinates = site_finder.find_adsorption_sites()
    if stats is not None:
        stats.site_finder_calls += 1
        stats.site_finding_seconds += perf_counter() - start
    return site_coordinates


def apply_adsorption_sites(
    cfg: Config,
    structure,
    adsorbate: Molecule,
    site_coordinates: dict[str, list],
) -> list[Structure]:
    site_finder = AdsorbateSiteFinder(slab=structure)
    structures = []
    for site in cfg.adsorbate.sites:
        for ads_coords in site_coordinates[site]:
            adsorbed_structure = site_finder.add_adsorbate(
                molecule=adsorbate, ads_coord=ads_coords
            )
            structures.append(adsorbed_structure)
    return structures


def get_top_layer_plane_height(structure: Structure, atoms_per_layer: int) -> float:
    top_layer_coords = structure.cart_coords[:atoms_per_layer]
    return float(np.mean(top_layer_coords[:, 2]))


def build_adsorption_site_template(
    structure: Structure,
    site_coordinates: dict[str, list],
    atoms_per_layer: int,
) -> AdsorptionSiteTemplate:
    surface_height = get_top_layer_plane_height(
        structure=structure, atoms_per_layer=atoms_per_layer
    )
    coordinates_by_site = {}
    for site, coordinates in site_coordinates.items():
        template_coordinates = []
        for coordinate in coordinates:
            fractional = structure.lattice.get_fractional_coords(coordinate)
            template_coordinates.append(
                (
                    float(fractional[0] % 1.0),
                    float(fractional[1] % 1.0),
                    float(coordinate[2] - surface_height),
                )
            )
        coordinates_by_site[site] = template_coordinates
    return AdsorptionSiteTemplate(coordinates_by_site=coordinates_by_site)


def transfer_adsorption_site_template(
    structure: Structure,
    template: AdsorptionSiteTemplate,
    atoms_per_layer: int,
) -> dict[str, list[np.ndarray]]:
    surface_height = get_top_layer_plane_height(
        structure=structure, atoms_per_layer=atoms_per_layer
    )
    transferred_coordinates = {}
    for site, coordinates in template.coordinates_by_site.items():
        site_coordinates = []
        for frac_x, frac_y, z_offset in coordinates:
            cartesian = structure.lattice.get_cartesian_coords([frac_x, frac_y, 0.0])
            site_coordinates.append(
                np.array([cartesian[0], cartesian[1], surface_height + z_offset])
            )
        transferred_coordinates[site] = site_coordinates
    return transferred_coordinates


def get_cached_adsorption_sites(
    cfg: Config,
    atoms: Atoms,
    structure: Structure,
    atoms_per_layer: int,
    motif_site_cache: dict[str, AdsorptionSiteTemplate],
    stats: AdsorbateGenerationStats | None = None,
) -> dict[str, list]:
    motif = classify_top_layer_motif(atoms=atoms, atoms_per_layer=atoms_per_layer)
    can_reuse = supports_two_swap_motif_template_reuse(cfg=cfg, motif=motif)
    if can_reuse and motif in motif_site_cache:
        return transfer_adsorption_site_template(
            structure=structure,
            template=motif_site_cache[motif],
            atoms_per_layer=atoms_per_layer,
        )

    site_coordinates = discover_adsorption_sites(structure=structure, stats=stats)
    if can_reuse:
        motif_site_cache[motif] = build_adsorption_site_template(
            structure=structure,
            site_coordinates=site_coordinates,
            atoms_per_layer=atoms_per_layer,
        )
    return site_coordinates


def get_adsorbate_indices(structure: Structure, adsorbate: Molecule) -> list[int]:
    """Returns the indices of the adsorbate in the structure."""
    structure_len = len(structure)
    adsorbate_len = len(adsorbate)
    adsorbate_indices = list(range(structure_len, structure_len + adsorbate_len))
    return adsorbate_indices


def get_adsorbate_comparison_indices(
    atoms_per_layer: int,
    surface_layers: int,
    adsorbate_indices: list[int],
    adsorbate_atoms_for_matching: int = 1,
) -> list[int]:
    """Returns surface and adsorbate indices used for structure matching."""
    surface_indices = list(range(atoms_per_layer * surface_layers))
    matched_adsorbate_indices = adsorbate_indices[:adsorbate_atoms_for_matching]
    return surface_indices + matched_adsorbate_indices


def get_adsorbate_structures(
    cfg: Config,
    atoms_list: list[Atoms],
    adsorbate: Molecule,
    matcher: StructureMatcher,
    converter: AseAtomsAdaptor = AseAtomsAdaptor(),
):
    """Generates all unique structures with the specified adsorbate."""
    source_atoms_list = atoms_list
    slabs = prepare_for_pymatgen(source_atoms_list)
    if not slabs:
        return []
    stats = AdsorbateGenerationStats(slabs_processed=len(slabs))
    atoms_per_layer = get_atoms_per_layer(cfg=cfg)
    comparison_indices = get_adsorbate_comparison_indices(
        atoms_per_layer=atoms_per_layer,
        surface_layers=cfg.adsorbate.surface_layers_for_matching,
        adsorbate_indices=get_adsorbate_indices(structure=slabs[0], adsorbate=adsorbate),
        adsorbate_atoms_for_matching=1,
    )
    motif_site_cache = {}
    atoms_list = []
    for atoms, slab in zip(source_atoms_list, slabs):
        site_coordinates = get_cached_adsorption_sites(
            cfg=cfg,
            atoms=atoms,
            structure=slab,
            atoms_per_layer=atoms_per_layer,
            motif_site_cache=motif_site_cache,
            stats=stats,
        )
        structures = apply_adsorption_sites(
            cfg=cfg,
            structure=slab,
            adsorbate=adsorbate,
            site_coordinates=site_coordinates,
        )
        matching_start = perf_counter()
        subsubstructures = [
            get_substructure(structure, indices=comparison_indices)
            for structure in structures
        ]
        unique_indices = find_unique_structures(
            structures=subsubstructures, matcher=matcher
        )
        stats.matching_seconds += perf_counter() - matching_start
        unique_structures = [structures[i] for i in unique_indices]
        unique_atoms = [converter.get_atoms(struct) for struct in unique_structures]
        atoms_list.extend(unique_atoms)
    print(stats.summary())
    return atoms_list
