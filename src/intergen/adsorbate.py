from dataclasses import dataclass
from copy import deepcopy
from time import perf_counter

import numpy as np
from ase import Atoms
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.analysis.structure_matcher import StructureMatcher
from intergen.config import Config
from intergen.metadata import (
    PARENT_SLAB_ID_KEY,
    SLAB_ID_KEY,
    SLAB_PROVENANCE_FIELDS,
)
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
    pymatgen_conversion_seconds: float = 0.0
    site_finding_seconds: float = 0.0
    site_selection_seconds: float = 0.0
    adsorbate_placement_seconds: float = 0.0
    matching_seconds: float = 0.0
    ase_conversion_seconds: float = 0.0

    def summary(self) -> str:
        return (
            "Adsorbate generation stats: "
            f"slabs={self.slabs_processed}, "
            f"site_finder_calls={self.site_finder_calls}, "
            f"pymatgen_conversion_seconds={self.pymatgen_conversion_seconds:.3f}, "
            f"site_finding_seconds={self.site_finding_seconds:.3f}, "
            f"site_selection_seconds={self.site_selection_seconds:.3f}, "
            f"adsorbate_placement_seconds={self.adsorbate_placement_seconds:.3f}, "
            f"matching_seconds={self.matching_seconds:.3f}, "
            f"ase_conversion_seconds={self.ase_conversion_seconds:.3f}"
        )


@dataclass
class AdsorbateSlabStats:
    slab_index: int
    cache_key: tuple[str, str]
    template_eligible: bool
    template_cache_hit: bool
    structures_emitted: int
    site_finder_calls: int
    pymatgen_conversion_seconds: float
    site_finding_seconds: float
    site_selection_seconds: float
    adsorbate_placement_seconds: float
    matching_seconds: float
    ase_conversion_seconds: float

    def summary(self) -> str:
        host_element, motif = self.cache_key
        path = "template-hit" if self.template_cache_hit else "seed-or-fallback"
        return (
            "Adsorbate slab stats: "
            f"slab={self.slab_index}, "
            f"host={host_element}, "
            f"motif={motif}, "
            f"template_eligible={self.template_eligible}, "
            f"path={path}, "
            f"structures={self.structures_emitted}, "
            f"site_finder_calls={self.site_finder_calls}, "
            f"pymatgen_conversion_seconds={self.pymatgen_conversion_seconds:.3f}, "
            f"site_finding_seconds={self.site_finding_seconds:.3f}, "
            f"site_selection_seconds={self.site_selection_seconds:.3f}, "
            f"adsorbate_placement_seconds={self.adsorbate_placement_seconds:.3f}, "
            f"matching_seconds={self.matching_seconds:.3f}, "
            f"ase_conversion_seconds={self.ase_conversion_seconds:.3f}"
        )


MOTIF_SITE_CACHEABLE = {"heterodimer", "dual_single_atom_alloy"}
DEFAULT_TEMPLATE_SITE_MATCH_TOLERANCE = 0.5


@dataclass
class AdsorptionSiteTemplate:
    coordinates_by_site: dict[str, list[tuple[float, float, float]]]


def get_top_layer_host_element(atoms: Atoms, atoms_per_layer: int) -> str:
    top_layer_symbols = atoms.get_chemical_symbols()[:atoms_per_layer]
    return max(set(top_layer_symbols), key=top_layer_symbols.count)


def build_adslab_provenance_metadata(parent_slab: Atoms) -> dict[str, object]:
    provenance = {
        key: deepcopy(parent_slab.info[key])
        for key in SLAB_PROVENANCE_FIELDS
        if key in parent_slab.info
    }
    if SLAB_ID_KEY in provenance:
        provenance[PARENT_SLAB_ID_KEY] = provenance.pop(SLAB_ID_KEY)
    return provenance


def attach_parent_slab_metadata(adslab: Atoms, parent_slab: Atoms) -> Atoms:
    adslab.info.update(build_adslab_provenance_metadata(parent_slab))
    return adslab


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
    motif_site_cache: dict[tuple[str, str], AdsorptionSiteTemplate],
    stats: AdsorbateGenerationStats | None = None,
) -> dict[str, list]:
    host_element = get_top_layer_host_element(
        atoms=atoms, atoms_per_layer=atoms_per_layer
    )
    motif = classify_top_layer_motif(atoms=atoms, atoms_per_layer=atoms_per_layer)
    cache_key = (host_element, motif)
    can_reuse = supports_two_swap_motif_template_reuse(cfg=cfg, motif=motif)
    if can_reuse and cache_key in motif_site_cache:
        return transfer_adsorption_site_template(
            structure=structure,
            template=motif_site_cache[cache_key],
            atoms_per_layer=atoms_per_layer,
        )

    site_coordinates = discover_adsorption_sites(structure=structure, stats=stats)
    if can_reuse:
        motif_site_cache[cache_key] = build_adsorption_site_template(
            structure=structure,
            site_coordinates=site_coordinates,
            atoms_per_layer=atoms_per_layer,
        )
    return site_coordinates


def resolve_adsorption_sites(
    cfg: Config,
    atoms: Atoms,
    structure: Structure,
    atoms_per_layer: int,
    motif_site_cache: dict[tuple[str, str], AdsorptionSiteTemplate],
    stats: AdsorbateGenerationStats | None = None,
) -> dict[str, list]:
    return get_cached_adsorption_sites(
        cfg=cfg,
        atoms=atoms,
        structure=structure,
        atoms_per_layer=atoms_per_layer,
        motif_site_cache=motif_site_cache,
        stats=stats,
    )


def get_adsorption_template_cache_details(
    cfg: Config,
    atoms: Atoms,
    atoms_per_layer: int,
) -> tuple[bool, tuple[str, str]]:
    host_element = get_top_layer_host_element(
        atoms=atoms, atoms_per_layer=atoms_per_layer
    )
    motif = classify_top_layer_motif(atoms=atoms, atoms_per_layer=atoms_per_layer)
    return supports_two_swap_motif_template_reuse(cfg=cfg, motif=motif), (
        host_element,
        motif,
    )


def build_adsorbate_slab_stats(
    slab_index: int,
    cache_key: tuple[str, str],
    template_eligible: bool,
    template_cache_hit: bool,
    structures_emitted: int,
    stats_before: AdsorbateGenerationStats,
    stats_after: AdsorbateGenerationStats,
) -> AdsorbateSlabStats:
    return AdsorbateSlabStats(
        slab_index=slab_index,
        cache_key=cache_key,
        template_eligible=template_eligible,
        template_cache_hit=template_cache_hit,
        structures_emitted=structures_emitted,
        site_finder_calls=stats_after.site_finder_calls - stats_before.site_finder_calls,
        pymatgen_conversion_seconds=(
            stats_after.pymatgen_conversion_seconds
            - stats_before.pymatgen_conversion_seconds
        ),
        site_finding_seconds=(
            stats_after.site_finding_seconds - stats_before.site_finding_seconds
        ),
        site_selection_seconds=(
            stats_after.site_selection_seconds - stats_before.site_selection_seconds
        ),
        adsorbate_placement_seconds=(
            stats_after.adsorbate_placement_seconds
            - stats_before.adsorbate_placement_seconds
        ),
        matching_seconds=stats_after.matching_seconds - stats_before.matching_seconds,
        ase_conversion_seconds=(
            stats_after.ase_conversion_seconds - stats_before.ase_conversion_seconds
        ),
    )


def get_site_coordinate_distance(
    coordinate_a: np.ndarray | tuple[float, float, float],
    coordinate_b: np.ndarray | tuple[float, float, float],
    lattice: Lattice | None = None,
) -> float:
    if lattice is not None:
        fractional_delta = lattice.get_fractional_coords(coordinate_a) - lattice.get_fractional_coords(
            coordinate_b
        )
        fractional_delta -= np.round(fractional_delta)
        cartesian_delta = lattice.get_cartesian_coords(fractional_delta)
        return float(np.linalg.norm(cartesian_delta))
    return float(np.linalg.norm(np.asarray(coordinate_a) - np.asarray(coordinate_b)))


def select_sites_matching_template_coordinates(
    discovered_coordinates: list[np.ndarray],
    template_coordinates: list[np.ndarray],
    tolerance: float = DEFAULT_TEMPLATE_SITE_MATCH_TOLERANCE,
    lattice: Lattice | None = None,
) -> list[np.ndarray]:
    # Explicit greedy policy: sort all valid template/discovered pairs by distance
    # and accept the nearest pair whenever neither side has been used yet.
    candidate_matches = []
    for template_index, template_coordinate in enumerate(template_coordinates):
        for discovered_index, discovered_coordinate in enumerate(discovered_coordinates):
            distance = get_site_coordinate_distance(
                template_coordinate, discovered_coordinate, lattice=lattice
            )
            if distance <= tolerance:
                candidate_matches.append((distance, template_index, discovered_index))

    candidate_matches.sort()
    used_template_indices = set()
    used_discovered_indices = set()
    selected_matches = []
    for _, template_index, discovered_index in candidate_matches:
        if template_index in used_template_indices:
            continue
        if discovered_index in used_discovered_indices:
            continue
        used_template_indices.add(template_index)
        used_discovered_indices.add(discovered_index)
        selected_matches.append((template_index, discovered_index))

    selected_matches.sort()
    return [
        discovered_coordinates[discovered_index]
        for _, discovered_index in selected_matches
    ]


def select_sites_matching_template(
    discovered_sites: dict[str, list[np.ndarray]],
    template_sites: dict[str, list[np.ndarray]],
    tolerance: float = DEFAULT_TEMPLATE_SITE_MATCH_TOLERANCE,
    lattice: Lattice | None = None,
) -> dict[str, list[np.ndarray]]:
    selected_sites = {}
    for site, template_coordinates in template_sites.items():
        selected_sites[site] = select_sites_matching_template_coordinates(
            discovered_coordinates=discovered_sites.get(site, []),
            template_coordinates=template_coordinates,
            tolerance=tolerance,
            lattice=lattice,
        )
    return selected_sites


def flatten_adsorption_site_coordinates(
    cfg: Config,
    site_coordinates: dict[str, list[np.ndarray]],
) -> list[tuple[str, np.ndarray]]:
    flattened_coordinates = []
    for site in cfg.adsorbate.sites:
        for coordinate in site_coordinates.get(site, []):
            flattened_coordinates.append((site, np.asarray(coordinate)))
    return flattened_coordinates


def group_adsorption_site_coordinates(
    flattened_coordinates: list[tuple[str, np.ndarray]],
) -> dict[str, list[np.ndarray]]:
    grouped_coordinates: dict[str, list[np.ndarray]] = {}
    for site, coordinate in flattened_coordinates:
        grouped_coordinates.setdefault(site, []).append(coordinate)
    return grouped_coordinates


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


def build_adsorbate_comparison_substructures(
    structures: list[Structure],
    comparison_indices: list[int],
) -> list[Structure]:
    return [
        get_substructure(structure, indices=comparison_indices)
        for structure in structures
    ]


def get_unique_adsorption_structure_indices(
    structures: list[Structure],
    comparison_indices: list[int],
    matcher: StructureMatcher,
    stats: AdsorbateGenerationStats | None = None,
) -> list[int]:
    matching_start = perf_counter()
    comparison_structures = build_adsorbate_comparison_substructures(
        structures=structures,
        comparison_indices=comparison_indices,
    )
    unique_indices = find_unique_structures(
        structures=comparison_structures, matcher=matcher
    )
    if stats is not None:
        stats.matching_seconds += perf_counter() - matching_start
    return unique_indices


def deduplicate_adsorption_structures(
    structures: list[Structure],
    comparison_indices: list[int],
    matcher: StructureMatcher,
    stats: AdsorbateGenerationStats | None = None,
) -> list[Structure]:
    unique_indices = get_unique_adsorption_structure_indices(
        structures=structures,
        comparison_indices=comparison_indices,
        matcher=matcher,
        stats=stats,
    )
    return [structures[i] for i in unique_indices]


def generate_adsorbate_structures_for_slab(
    cfg: Config,
    atoms: Atoms,
    slab: Structure,
    adsorbate: Molecule,
    atoms_per_layer: int,
    comparison_indices: list[int],
    matcher: StructureMatcher,
    motif_site_cache: dict[tuple[str, str], AdsorptionSiteTemplate],
    stats: AdsorbateGenerationStats | None = None,
) -> list[Structure]:
    can_reuse, cache_key = get_adsorption_template_cache_details(
        cfg=cfg,
        atoms=atoms,
        atoms_per_layer=atoms_per_layer,
    )
    discovered_sites = discover_adsorption_sites(structure=slab, stats=stats)

    if can_reuse and cache_key in motif_site_cache:
        template_sites = transfer_adsorption_site_template(
            structure=slab,
            template=motif_site_cache[cache_key],
            atoms_per_layer=atoms_per_layer,
        )
        # Cache hits treat the template as the canonical deduplicated site set.
        site_selection_start = perf_counter()
        selected_sites = select_sites_matching_template(
            discovered_sites=discovered_sites,
            template_sites=template_sites,
            tolerance=cfg.adsorbate.template_site_match_tolerance,
            lattice=slab.lattice,
        )
        if stats is not None:
            stats.site_selection_seconds += perf_counter() - site_selection_start
        placement_start = perf_counter()
        structures = apply_adsorption_sites(
            cfg=cfg,
            structure=slab,
            adsorbate=adsorbate,
            site_coordinates=selected_sites,
        )
        if stats is not None:
            stats.adsorbate_placement_seconds += perf_counter() - placement_start
        return structures

    placement_start = perf_counter()
    structures = apply_adsorption_sites(
        cfg=cfg,
        structure=slab,
        adsorbate=adsorbate,
        site_coordinates=discovered_sites,
    )
    if stats is not None:
        stats.adsorbate_placement_seconds += perf_counter() - placement_start
    unique_indices = get_unique_adsorption_structure_indices(
        structures=structures,
        comparison_indices=comparison_indices,
        matcher=matcher,
        stats=stats,
    )
    if can_reuse:
        flattened_coordinates = flatten_adsorption_site_coordinates(
            cfg=cfg,
            site_coordinates=discovered_sites,
        )
        unique_site_coordinates = group_adsorption_site_coordinates(
            [flattened_coordinates[index] for index in unique_indices]
        )
        motif_site_cache[cache_key] = build_adsorption_site_template(
            structure=slab,
            site_coordinates=unique_site_coordinates,
            atoms_per_layer=atoms_per_layer,
        )
    return [structures[i] for i in unique_indices]


def get_adsorbate_structures(
    cfg: Config,
    atoms_list: list[Atoms],
    adsorbate: Molecule,
    matcher: StructureMatcher,
    converter: AseAtomsAdaptor = AseAtomsAdaptor(),
):
    """Generates all unique structures with the specified adsorbate."""
    source_atoms_list = atoms_list
    conversion_start = perf_counter()
    slabs = prepare_for_pymatgen(source_atoms_list)
    if not slabs:
        return []
    stats = AdsorbateGenerationStats(slabs_processed=len(slabs))
    stats.pymatgen_conversion_seconds += perf_counter() - conversion_start
    atoms_per_layer = get_atoms_per_layer(cfg=cfg)
    comparison_indices = get_adsorbate_comparison_indices(
        atoms_per_layer=atoms_per_layer,
        surface_layers=cfg.adsorbate.surface_layers_for_matching,
        adsorbate_indices=get_adsorbate_indices(structure=slabs[0], adsorbate=adsorbate),
        adsorbate_atoms_for_matching=1,
    )
    motif_site_cache = {}
    atoms_list = []
    for slab_index, (atoms, slab) in enumerate(zip(source_atoms_list, slabs), start=1):
        template_eligible, cache_key = get_adsorption_template_cache_details(
            cfg=cfg,
            atoms=atoms,
            atoms_per_layer=atoms_per_layer,
        )
        template_cache_hit = template_eligible and cache_key in motif_site_cache
        stats_before = AdsorbateGenerationStats(
            slabs_processed=stats.slabs_processed,
            site_finder_calls=stats.site_finder_calls,
            pymatgen_conversion_seconds=stats.pymatgen_conversion_seconds,
            site_finding_seconds=stats.site_finding_seconds,
            site_selection_seconds=stats.site_selection_seconds,
            adsorbate_placement_seconds=stats.adsorbate_placement_seconds,
            matching_seconds=stats.matching_seconds,
            ase_conversion_seconds=stats.ase_conversion_seconds,
        )
        unique_structures = generate_adsorbate_structures_for_slab(
            cfg=cfg,
            atoms=atoms,
            slab=slab,
            adsorbate=adsorbate,
            atoms_per_layer=atoms_per_layer,
            comparison_indices=comparison_indices,
            matcher=matcher,
            motif_site_cache=motif_site_cache,
            stats=stats,
        )
        ase_conversion_start = perf_counter()
        unique_atoms = [
            attach_parent_slab_metadata(
                adslab=converter.get_atoms(struct),
                parent_slab=atoms,
            )
            for struct in unique_structures
        ]
        stats.ase_conversion_seconds += perf_counter() - ase_conversion_start
        slab_stats = build_adsorbate_slab_stats(
            slab_index=slab_index,
            cache_key=cache_key,
            template_eligible=template_eligible,
            template_cache_hit=template_cache_hit,
            structures_emitted=len(unique_structures),
            stats_before=stats_before,
            stats_after=stats,
        )
        print(slab_stats.summary())
        atoms_list.extend(unique_atoms)
    print(stats.summary())
    return atoms_list
