import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
from ase.build import fcc111
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Molecule
from pymatgen.io.ase import AseAtomsAdaptor

from intergen.adsorbate import (
    AdsorbedStructureRecord,
    AdsorbateGenerationStats,
    build_adsorption_site_template,
    build_adsorbate_metadata,
    build_adslab_provenance_metadata,
    add_adsorbates,
    apply_adsorption_sites,
    build_adsorbate_comparison_substructures,
    deduplicate_adsorption_structures,
    discover_adsorption_sites,
    generate_adsorbate_structures_for_slab,
    generate_adsorbate_structure_records_for_slab,
    get_site_coordinate_distance,
    get_adsorbate_comparison_indices,
    get_adsorbate_structures,
    get_top_layer_host_element,
    select_sites_matching_template,
    select_sites_matching_template_coordinates,
    supports_two_swap_motif_template_reuse,
    transfer_adsorption_site_template,
)
from intergen.config import Config
from intergen.metadata import (
    ADSLAB_ID_KEY,
    ADSORBATE_KEY,
    HOST_ELEMENT_KEY,
    INITIAL_SITE_COORDINATE_KEY,
    INITIAL_SITE_LABEL_KEY,
    PARENT_SLAB_ID_KEY,
    SLAB_ID_KEY,
    SURFACE_TYPE_FCC111,
    SURFACE_TYPE_KEY,
    SUPERCELL_SIZE_KEY,
    SWAP_ELEMENTS_KEY,
    SWAP_INDICES_KEY,
    TOP_LAYER_MOTIF_KEY,
    TOP_LAYER_MOTIF_SINGLE_SWAP,
)
from intergen.surface import (
    assign_slab_metadata,
    classify_top_layer_motif,
    find_unique_structures,
    get_substructure,
    swap_atoms,
)


def make_config(
    surface_layers_for_matching,
    reuse_site_templates_for_two_swap_motifs=True,
    num_swaps=1,
    size=(3, 3, 4),
    template_site_match_tolerance=0.5,
):
    return Config(
        structure={
            "bcc_list": [],
            "hcp_list": [],
            "fcc_list": ["Pt"],
            "size": size,
            "vacuum": 10.0,
        },
        generation={
            "layers_to_swap": 1,
            "num_swaps": num_swaps,
            "swap_elements": ["Cu"],
            "only_last_generation": True,
        },
        database={"path": "data/test.db"},
        adsorbate={
            "matcher": {"ltol": 0.05, "stol": 0.1, "angle_tol": 5.0},
            "species": "N",
            "coords": [(0.0, 0.0, 0.0)],
            "sites": ["hollow"],
            "tag": 0,
            "surface_layers_for_matching": surface_layers_for_matching,
            "reuse_site_templates_for_two_swap_motifs": (
                reuse_site_templates_for_two_swap_motifs
            ),
            "template_site_match_tolerance": template_site_match_tolerance,
        },
    )


class TestConfig(unittest.TestCase):
    def test_adsorbate_surface_layers_for_matching_is_parsed(self):
        cfg = make_config(surface_layers_for_matching=2)

        self.assertEqual(cfg.adsorbate.surface_layers_for_matching, 2)

    def test_adsorbate_reuse_site_templates_flag_is_parsed(self):
        cfg = make_config(
            surface_layers_for_matching=2,
            reuse_site_templates_for_two_swap_motifs=False,
        )

        self.assertFalse(cfg.adsorbate.reuse_site_templates_for_two_swap_motifs)

    def test_adsorbate_template_site_match_tolerance_is_parsed(self):
        cfg = make_config(
            surface_layers_for_matching=2,
            template_site_match_tolerance=0.25,
        )

        self.assertEqual(cfg.adsorbate.template_site_match_tolerance, 0.25)

    def test_supports_two_swap_motif_template_reuse_requires_supported_case(self):
        supported_cfg = make_config(
            surface_layers_for_matching=2,
            reuse_site_templates_for_two_swap_motifs=True,
            num_swaps=2,
        )
        wrong_swap_count_cfg = make_config(
            surface_layers_for_matching=2,
            reuse_site_templates_for_two_swap_motifs=True,
            num_swaps=1,
        )
        wrong_size_cfg = make_config(
            surface_layers_for_matching=2,
            reuse_site_templates_for_two_swap_motifs=True,
            num_swaps=2,
            size=(4, 3, 4),
        )

        self.assertTrue(
            supports_two_swap_motif_template_reuse(
                cfg=supported_cfg, motif="heterodimer"
            )
        )
        self.assertFalse(
            supports_two_swap_motif_template_reuse(
                cfg=wrong_swap_count_cfg, motif="heterodimer"
            )
        )
        self.assertFalse(
            supports_two_swap_motif_template_reuse(
                cfg=wrong_size_cfg, motif="heterodimer"
            )
        )
        self.assertFalse(
            supports_two_swap_motif_template_reuse(
                cfg=supported_cfg, motif="single_swap"
            )
        )

    def test_top_layer_host_element_uses_majority_species(self):
        atoms = fcc111("Pt", size=(3, 3, 4), vacuum=10.0)[::-1]
        atoms = swap_atoms(atoms, 0, "Cu")
        atoms = swap_atoms(atoms, 1, "Au")

        self.assertEqual(get_top_layer_host_element(atoms, atoms_per_layer=9), "Pt")


class TestAdsorbateMetadata(unittest.TestCase):
    def test_build_adsorbate_metadata_normalizes_site_fields(self):
        adsorbate = Molecule(
            ["O", "H"],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.9]],
            site_properties={"tags": [0, 0]},
        )

        metadata = build_adsorbate_metadata(
            adsorbate=adsorbate,
            site_label="fcc hollow",
            site_coordinate=np.array([1, 2.5, 3]),
            adslab_id="adslab-000001",
        )

        self.assertEqual(metadata[ADSLAB_ID_KEY], "adslab-000001")
        self.assertEqual(metadata[ADSORBATE_KEY], "OH")
        self.assertEqual(metadata[INITIAL_SITE_LABEL_KEY], "fcc_hollow")
        self.assertEqual(metadata[INITIAL_SITE_COORDINATE_KEY], (1.0, 2.5, 3.0))

    def test_build_adslab_provenance_metadata_renames_slab_id_to_parent_slab_id(self):
        slab = fcc111("Pt", size=(2, 2, 3), vacuum=10.0)[::-1]
        assign_slab_metadata(
            slab,
            slab_id="slab-000010",
            host_element="Pt",
            surface_type=SURFACE_TYPE_FCC111,
            supercell_size=(2, 2, 3),
            swap_indices=[0],
            swap_elements=["Cu"],
            top_layer_motif=TOP_LAYER_MOTIF_SINGLE_SWAP,
        )

        metadata = build_adslab_provenance_metadata(slab)

        self.assertNotIn(SLAB_ID_KEY, metadata)
        self.assertEqual(metadata[PARENT_SLAB_ID_KEY], "slab-000010")
        self.assertEqual(metadata[HOST_ELEMENT_KEY], "Pt")
        self.assertEqual(metadata[SURFACE_TYPE_KEY], SURFACE_TYPE_FCC111)
        self.assertEqual(metadata[SUPERCELL_SIZE_KEY], (2, 2, 3))
        self.assertEqual(metadata[SWAP_INDICES_KEY], [0])
        self.assertEqual(metadata[SWAP_ELEMENTS_KEY], ["Cu"])
        self.assertEqual(metadata[TOP_LAYER_MOTIF_KEY], TOP_LAYER_MOTIF_SINGLE_SWAP)

    def test_get_adsorbate_structures_preserves_parent_slab_provenance(self):
        cfg = make_config(surface_layers_for_matching=2)
        slab = fcc111("Pt", size=(3, 3, 4), vacuum=10.0)[::-1]
        slab = swap_atoms(slab, 0, "Cu")
        assign_slab_metadata(
            slab,
            slab_id="slab-000020",
            host_element="Pt",
            surface_type=SURFACE_TYPE_FCC111,
            supercell_size=(3, 3, 4),
            swap_indices=[0],
            swap_elements=["Cu"],
            top_layer_motif=TOP_LAYER_MOTIF_SINGLE_SWAP,
        )
        slab.set_pbc(True)
        adsorbate = Molecule(
            ["N"],
            [[0.0, 0.0, 0.0]],
            site_properties={"tags": [0]},
        )
        matcher = StructureMatcher(**cfg.adsorbate.matcher.model_dump())
        with patch(
            "intergen.adsorbate.generate_adsorbate_structure_records_for_slab",
            return_value=[
                AdsorbedStructureRecord(
                    structure=AseAtomsAdaptor().get_structure(slab),
                    site_label="hollow",
                    site_coordinate=(1.0, 2.0, 3.0),
                )
            ],
        ):
            adslabs = get_adsorbate_structures(
                cfg=cfg,
                atoms_list=[slab],
                adsorbate=adsorbate,
                matcher=matcher,
            )

        self.assertEqual(len(adslabs), 1)
        self.assertNotIn(SLAB_ID_KEY, adslabs[0].info)
        self.assertEqual(adslabs[0].info[ADSLAB_ID_KEY], "adslab-000001")
        self.assertEqual(adslabs[0].info[PARENT_SLAB_ID_KEY], "slab-000020")
        self.assertEqual(adslabs[0].info[ADSORBATE_KEY], "N")
        self.assertEqual(adslabs[0].info[INITIAL_SITE_LABEL_KEY], "hollow")
        self.assertEqual(adslabs[0].info[INITIAL_SITE_COORDINATE_KEY], (1.0, 2.0, 3.0))
        self.assertEqual(adslabs[0].info[HOST_ELEMENT_KEY], "Pt")
        self.assertEqual(adslabs[0].info[SURFACE_TYPE_KEY], SURFACE_TYPE_FCC111)
        self.assertEqual(adslabs[0].info[SUPERCELL_SIZE_KEY], (3, 3, 4))
        self.assertEqual(adslabs[0].info[SWAP_INDICES_KEY], [0])
        self.assertEqual(adslabs[0].info[SWAP_ELEMENTS_KEY], ["Cu"])
        self.assertEqual(adslabs[0].info[TOP_LAYER_MOTIF_KEY], TOP_LAYER_MOTIF_SINGLE_SWAP)

    def test_generate_adsorbate_structure_records_for_slab_keeps_site_metadata(self):
        cfg = make_config(surface_layers_for_matching=2)
        slab_atoms = fcc111("Pt", size=(3, 3, 4), vacuum=10.0)[::-1]
        slab_structure = AseAtomsAdaptor().get_structure(slab_atoms)
        adsorbate = Molecule(
            ["N"],
            [[0.0, 0.0, 0.0]],
            site_properties={"tags": [0]},
        )
        matcher = StructureMatcher(**cfg.adsorbate.matcher.model_dump())

        with patch(
            "intergen.adsorbate.discover_adsorption_sites",
            return_value={"hollow": [np.array([0.1, 0.2, 0.3])]},
        ):
            records = generate_adsorbate_structure_records_for_slab(
                cfg=cfg,
                atoms=slab_atoms,
                slab=slab_structure,
                adsorbate=adsorbate,
                atoms_per_layer=9,
                comparison_indices=list(range(19)),
                matcher=matcher,
                motif_site_cache={},
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].site_label, "hollow")
        self.assertEqual(records[0].site_coordinate, (0.1, 0.2, 0.3))

    def test_get_adsorbate_structures_assigns_unique_adslab_ids_within_run(self):
        cfg = make_config(surface_layers_for_matching=2)
        adsorbate = Molecule(
            ["N"],
            [[0.0, 0.0, 0.0]],
            site_properties={"tags": [0]},
        )
        matcher = StructureMatcher(**cfg.adsorbate.matcher.model_dump())

        slab_a = fcc111("Pt", size=(3, 3, 4), vacuum=10.0)[::-1]
        assign_slab_metadata(
            slab_a,
            slab_id="slab-000001",
            host_element="Pt",
            surface_type=SURFACE_TYPE_FCC111,
            supercell_size=(3, 3, 4),
            top_layer_motif="pure",
        )
        slab_b = fcc111("Pt", size=(3, 3, 4), vacuum=10.0)[::-1]
        assign_slab_metadata(
            slab_b,
            slab_id="slab-000002",
            host_element="Pt",
            surface_type=SURFACE_TYPE_FCC111,
            supercell_size=(3, 3, 4),
            top_layer_motif="pure",
        )

        record_a = AdsorbedStructureRecord(
            structure=AseAtomsAdaptor().get_structure(slab_a),
            site_label="top",
            site_coordinate=(0.0, 0.0, 1.0),
        )
        record_b = AdsorbedStructureRecord(
            structure=AseAtomsAdaptor().get_structure(slab_b),
            site_label="bridge",
            site_coordinate=(1.0, 0.0, 1.0),
        )

        with patch(
            "intergen.adsorbate.generate_adsorbate_structure_records_for_slab",
            side_effect=[[record_a], [record_b]],
        ):
            adslabs = get_adsorbate_structures(
                cfg=cfg,
                atoms_list=[slab_a, slab_b],
                adsorbate=adsorbate,
                matcher=matcher,
            )

        self.assertEqual(
            [adslab.info[ADSLAB_ID_KEY] for adslab in adslabs],
            ["adslab-000001", "adslab-000002"],
        )


class TestTemplateSiteMatching(unittest.TestCase):
    def test_get_site_coordinate_distance_returns_euclidean_distance(self):
        distance = get_site_coordinate_distance(
            np.array([0.0, 0.0, 0.0]),
            np.array([3.0, 4.0, 0.0]),
        )

        self.assertEqual(distance, 5.0)

    def test_get_site_coordinate_distance_uses_minimum_image_with_lattice(self):
        distance = get_site_coordinate_distance(
            np.array([0.0, 0.0, 1.0]),
            np.array([9.8, 0.0, 1.0]),
            lattice=Lattice.cubic(10.0),
        )

        self.assertAlmostEqual(distance, 0.2)

    def test_select_sites_matching_template_coordinates_keeps_closest_match(self):
        discovered_coordinates = [
            np.array([0.04, 0.0, 1.0]),
            np.array([0.18, 0.0, 1.0]),
            np.array([1.07, 0.0, 1.0]),
        ]
        template_coordinates = [
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 1.0]),
        ]

        selected_coordinates = select_sites_matching_template_coordinates(
            discovered_coordinates=discovered_coordinates,
            template_coordinates=template_coordinates,
            tolerance=0.2,
        )

        self.assertEqual(len(selected_coordinates), 2)
        self.assertTrue(np.allclose(selected_coordinates[0], discovered_coordinates[0]))
        self.assertTrue(np.allclose(selected_coordinates[1], discovered_coordinates[2]))

    def test_select_sites_matching_template_coordinates_respects_tolerance(self):
        discovered_coordinates = [
            np.array([0.25, 0.0, 1.0]),
            np.array([1.4, 0.0, 1.0]),
        ]
        template_coordinates = [
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 1.0]),
        ]

        selected_coordinates = select_sites_matching_template_coordinates(
            discovered_coordinates=discovered_coordinates,
            template_coordinates=template_coordinates,
            tolerance=0.2,
        )

        self.assertEqual(selected_coordinates, [])

    def test_select_sites_matching_template_coordinates_uses_explicit_greedy_policy(
        self,
    ):
        shared_discovered_coordinate = np.array([0.08, 0.0, 1.0])
        discovered_coordinates = [
            shared_discovered_coordinate,
            np.array([0.22, 0.0, 1.0]),
        ]
        template_coordinates = [
            np.array([0.0, 0.0, 1.0]),
            np.array([0.15, 0.0, 1.0]),
        ]

        selected_coordinates = select_sites_matching_template_coordinates(
            discovered_coordinates=discovered_coordinates,
            template_coordinates=template_coordinates,
            tolerance=0.2,
        )

        # The nearest pair wins first, even though a different global assignment
        # could match both template sites.
        self.assertEqual(len(selected_coordinates), 1)
        self.assertTrue(
            np.allclose(selected_coordinates[0], shared_discovered_coordinate)
        )

    def test_select_sites_matching_template_matches_per_site_name(self):
        discovered_sites = {
            "bridge": [np.array([0.03, 0.0, 1.0]), np.array([2.0, 0.0, 1.0])],
            "hollow": [np.array([1.02, 0.0, 1.0])],
        }
        template_sites = {
            "bridge": [np.array([0.0, 0.0, 1.0])],
            "hollow": [np.array([1.0, 0.0, 1.0])],
        }

        selected_sites = select_sites_matching_template(
            discovered_sites=discovered_sites,
            template_sites=template_sites,
            tolerance=0.1,
        )

        self.assertEqual(set(selected_sites), {"bridge", "hollow"})
        self.assertEqual(len(selected_sites["bridge"]), 1)
        self.assertEqual(len(selected_sites["hollow"]), 1)
        self.assertTrue(
            np.allclose(selected_sites["bridge"][0], discovered_sites["bridge"][0])
        )
        self.assertTrue(
            np.allclose(selected_sites["hollow"][0], discovered_sites["hollow"][0])
        )

    def test_select_sites_matching_template_coordinates_matches_across_boundary(self):
        discovered_coordinates = [np.array([9.8, 0.0, 1.0])]
        template_coordinates = [np.array([0.0, 0.0, 1.0])]

        selected_coordinates = select_sites_matching_template_coordinates(
            discovered_coordinates=discovered_coordinates,
            template_coordinates=template_coordinates,
            tolerance=0.2,
            lattice=Lattice.cubic(10.0),
        )

        self.assertEqual(len(selected_coordinates), 1)
        self.assertTrue(np.allclose(selected_coordinates[0], discovered_coordinates[0]))


class TestHollowSiteRegistry(unittest.TestCase):
    class CountingMatcher:
        def __init__(self, matcher):
            self.matcher = matcher
            self.fit_calls = 0

        def fit(self, left_structure, right_structure):
            self.fit_calls += 1
            return self.matcher.fit(left_structure, right_structure)

    @staticmethod
    def _make_host_atoms(host, size=(3, 3, 4)):
        atoms = fcc111(host, size=size, vacuum=10.0)[::-1]
        atoms.set_tags([0] * len(atoms))
        atoms.set_pbc(True)
        return atoms

    @classmethod
    def setUpClass(cls):
        cls.cfg = make_config(surface_layers_for_matching=2)
        cls.atoms = cls._make_host_atoms("Pt")
        cls.converter = AseAtomsAdaptor()
        cls.slab = cls.converter.get_structure(cls.atoms)
        cls.site_finder = AdsorbateSiteFinder(cls.slab)
        cls.adsorbate = Molecule(
            ["N"], [[0.0, 0.0, 0.0]], site_properties={"tags": [0]}
        )
        cls.matcher = StructureMatcher(ltol=0.05, stol=0.1, angle_tol=5.0)
        cls.atoms_per_layer = 9
        cls.adsorbate_index = [len(cls.slab)]
        cls.comparison_indices = get_adsorbate_comparison_indices(
            atoms_per_layer=cls.atoms_per_layer,
            surface_layers=cls.cfg.adsorbate.surface_layers_for_matching,
            adsorbate_indices=cls.adsorbate_index,
        )
        cls.discovered_sites = discover_adsorption_sites(cls.slab)
        cls.split_structures = apply_adsorption_sites(
            cfg=cls.cfg,
            structure=cls.slab,
            adsorbate=cls.adsorbate,
            site_coordinates=cls.discovered_sites,
        )
        cls.hollow_sites = cls.site_finder.find_adsorption_sites(symm_reduce=False)[
            "hollow"
        ]
        cls.hollow_structures = [
            cls.site_finder.add_adsorbate(cls.adsorbate, coords)
            for coords in cls.hollow_sites
        ]

    def _comparison_substructures(self, structures, surface_layers):
        comparison_indices = get_adsorbate_comparison_indices(
            atoms_per_layer=self.atoms_per_layer,
            surface_layers=surface_layers,
            adsorbate_indices=self.adsorbate_index,
        )
        return [
            get_substructure(structure, indices=comparison_indices)
            for structure in structures
        ]

    def _canonical_structure_signature(self, structure):
        species = tuple(str(site.specie) for site in structure)
        lattice = tuple(
            tuple(round(float(value), 6) for value in row)
            for row in structure.lattice.matrix
        )
        fractional_coordinates = tuple(
            tuple(round(float(value % 1.0), 6) for value in coordinate)
            for coordinate in structure.frac_coords
        )
        return species, lattice, fractional_coordinates

    def _canonical_structure_signatures(self, structures):
        return sorted(
            self._canonical_structure_signature(structure) for structure in structures
        )

    def _canonical_atom_signatures(self, atoms_list):
        structures = [self.converter.get_structure(atoms) for atoms in atoms_list]
        return self._canonical_structure_signatures(structures)

    def _get_unique_adsorbate_structures(self, structures):
        return deduplicate_adsorption_structures(
            structures=structures,
            comparison_indices=self.comparison_indices,
            matcher=self.matcher,
        )

    def _get_unique_adsorbate_substructure_signatures(self, structures):
        unique_structures = self._get_unique_adsorbate_structures(structures)
        comparison_substructures = build_adsorbate_comparison_substructures(
            structures=unique_structures,
            comparison_indices=self.comparison_indices,
        )
        return self._canonical_structure_signatures(comparison_substructures)

    def _get_comparison_substructure_signatures(self, structures, surface_layers):
        return self._canonical_structure_signatures(
            self._comparison_substructures(
                structures,
                surface_layers=surface_layers,
            )
        )

    def _get_unique_comparison_substructure_indices(self, structures, surface_layers):
        return find_unique_structures(
            self._comparison_substructures(
                structures,
                surface_layers=surface_layers,
            ),
            matcher=self.matcher,
        )

    def _assert_structure_lists_match(
        self, left_structures, right_structures, matcher=None
    ):
        del matcher
        self.assertEqual(
            self._canonical_atom_signatures(left_structures),
            self._canonical_atom_signatures(right_structures),
        )

    def _assert_sites_match(self, transferred_sites, direct_sites, site_name):
        self.assertEqual(
            len(transferred_sites[site_name]),
            len(direct_sites[site_name]),
        )
        for transferred, direct in zip(
            transferred_sites[site_name],
            direct_sites[site_name],
        ):
            self.assertTrue(np.allclose(transferred, direct, atol=1e-6))

    def _assert_transferred_path_matches_direct_path(
        self,
        reference_atoms,
        target_atoms,
        site_name,
    ):
        reference_structure = AseAtomsAdaptor().get_structure(reference_atoms)
        target_structure = AseAtomsAdaptor().get_structure(target_atoms)
        reference_sites = discover_adsorption_sites(reference_structure)
        direct_sites = discover_adsorption_sites(target_structure)
        template = build_adsorption_site_template(
            structure=reference_structure,
            site_coordinates=reference_sites,
            atoms_per_layer=self.atoms_per_layer,
        )
        transferred_sites = transfer_adsorption_site_template(
            structure=target_structure,
            template=template,
            atoms_per_layer=self.atoms_per_layer,
        )

        self._assert_sites_match(transferred_sites, direct_sites, site_name)

        direct_structures = apply_adsorption_sites(
            cfg=self.cfg,
            structure=target_structure,
            adsorbate=self.adsorbate,
            site_coordinates=direct_sites,
        )
        transferred_structures = apply_adsorption_sites(
            cfg=self.cfg,
            structure=target_structure,
            adsorbate=self.adsorbate,
            site_coordinates=transferred_sites,
        )
        unique_direct_structures = self._get_unique_adsorbate_structures(
            direct_structures
        )
        unique_transferred_structures = self._get_unique_adsorbate_structures(
            transferred_structures
        )

        self.assertEqual(
            self._canonical_structure_signatures(unique_transferred_structures),
            self._canonical_structure_signatures(unique_direct_structures),
        )

    def _generate_structures_for_atoms_list(self, cfg, atoms_list, matcher=None):
        if matcher is None:
            matcher = self.matcher
        motif_site_cache = {}
        stats = AdsorbateGenerationStats(slabs_processed=len(atoms_list))
        generated_structures = []
        for atoms in atoms_list:
            structure = AseAtomsAdaptor().get_structure(atoms)
            generated_structures.extend(
                generate_adsorbate_structures_for_slab(
                    cfg=cfg,
                    atoms=atoms,
                    slab=structure,
                    adsorbate=self.adsorbate,
                    atoms_per_layer=self.atoms_per_layer,
                    comparison_indices=self.comparison_indices,
                    matcher=matcher,
                    motif_site_cache=motif_site_cache,
                    stats=stats,
                )
            )
        return generated_structures, stats, motif_site_cache

    def test_get_adsorbate_comparison_indices_uses_requested_surface_layers(self):
        comparison_indices = get_adsorbate_comparison_indices(
            atoms_per_layer=9,
            surface_layers=2,
            adsorbate_indices=[36, 37],
        )

        self.assertEqual(comparison_indices, list(range(18)) + [36])

    def test_get_adsorbate_comparison_indices_can_include_multiple_adsorbate_atoms(self):
        comparison_indices = get_adsorbate_comparison_indices(
            atoms_per_layer=9,
            surface_layers=2,
            adsorbate_indices=[36, 37],
            adsorbate_atoms_for_matching=2,
        )

        self.assertEqual(comparison_indices, list(range(18)) + [36, 37])

    def test_hollow_site_matching_distinguishes_fcc_and_hcp_with_second_layer(self):
        representative_indices = self._get_unique_comparison_substructure_indices(
            self.hollow_structures, surface_layers=2
        )
        representative_structures = [
            self.hollow_structures[index] for index in representative_indices
        ]

        self.assertEqual(len(representative_structures), 2)

        one_layer_unique = self._get_unique_comparison_substructure_indices(
            representative_structures, surface_layers=1
        )
        two_layer_unique = self._get_unique_comparison_substructure_indices(
            representative_structures, surface_layers=2
        )

        self.assertEqual(one_layer_unique, [0])
        self.assertEqual(two_layer_unique, [0, 1])

    def test_get_adsorbate_structures_distinguishes_hollow_registry_by_layer_count(self):
        one_layer_indices = get_adsorbate_comparison_indices(
            atoms_per_layer=self.atoms_per_layer,
            surface_layers=1,
            adsorbate_indices=self.adsorbate_index,
        )
        two_layer_indices = get_adsorbate_comparison_indices(
            atoms_per_layer=self.atoms_per_layer,
            surface_layers=2,
            adsorbate_indices=self.adsorbate_index,
        )

        one_layer_structures = deduplicate_adsorption_structures(
            structures=self.split_structures,
            comparison_indices=one_layer_indices,
            matcher=self.matcher,
        )
        two_layer_structures = deduplicate_adsorption_structures(
            structures=self.split_structures,
            comparison_indices=two_layer_indices,
            matcher=self.matcher,
        )

        self.assertEqual(len(one_layer_structures), 1)
        self.assertEqual(len(two_layer_structures), 2)

    def test_get_adsorbate_structures_prints_generation_stats(self):
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            structures = get_adsorbate_structures(
                cfg=self.cfg,
                atoms_list=[self.atoms],
                adsorbate=self.adsorbate,
                matcher=self.matcher,
            )

        output = stdout.getvalue()

        self.assertEqual(len(structures), 2)
        self.assertIn("Adsorbate generation stats:", output)
        self.assertIn("slabs=1", output)
        self.assertIn("site_finder_calls=1", output)

    def test_discover_then_apply_matches_add_adsorbates_for_single_slab(self):
        direct_structures = add_adsorbates(
            cfg=self.cfg,
            structure=self.slab,
            adsorbate=self.adsorbate,
        )

        self.assertEqual(len(self.split_structures), len(direct_structures))
        self.assertEqual(
            self._canonical_structure_signatures(self.split_structures),
            self._canonical_structure_signatures(direct_structures),
        )

    def test_build_adsorbate_comparison_substructures_matches_existing_helper(self):
        substructures = build_adsorbate_comparison_substructures(
            structures=self.split_structures,
            comparison_indices=self.comparison_indices,
        )
        expected_substructures = self._comparison_substructures(
            self.split_structures,
            surface_layers=self.cfg.adsorbate.surface_layers_for_matching,
        )

        self.assertEqual(
            self._canonical_structure_signatures(substructures),
            self._canonical_structure_signatures(expected_substructures),
        )

    def test_deduplicate_adsorption_structures_matches_existing_helper(self):
        deduplicated_structures = deduplicate_adsorption_structures(
            structures=self.split_structures,
            comparison_indices=self.comparison_indices,
            matcher=self.matcher,
        )
        expected_structures = self._get_unique_adsorbate_structures(self.split_structures)

        self.assertEqual(
            self._canonical_structure_signatures(deduplicated_structures),
            self._canonical_structure_signatures(expected_structures),
        )

    def test_generate_adsorbate_structures_for_slab_matches_split_workflow(self):
        motif_site_cache = {}
        expected_structures = deduplicate_adsorption_structures(
            structures=self.split_structures,
            comparison_indices=self.comparison_indices,
            matcher=self.matcher,
        )

        slab_structures = generate_adsorbate_structures_for_slab(
            cfg=self.cfg,
            atoms=self.atoms,
            slab=self.slab,
            adsorbate=self.adsorbate,
            atoms_per_layer=self.atoms_per_layer,
            comparison_indices=self.comparison_indices,
            matcher=self.matcher,
            motif_site_cache=motif_site_cache,
        )

        self.assertEqual(
            self._canonical_structure_signatures(slab_structures),
            self._canonical_structure_signatures(expected_structures),
        )

    def test_equivalent_heterodimer_slabs_reuse_cached_site_discovery(self):
        first_heterodimer = swap_atoms(self.atoms, 0, "Cu")
        first_heterodimer = swap_atoms(first_heterodimer, 1, "Au")
        second_heterodimer = swap_atoms(self.atoms, 3, "Cu")
        second_heterodimer = swap_atoms(second_heterodimer, 4, "Au")
        supported_cfg = make_config(
            surface_layers_for_matching=2,
            reuse_site_templates_for_two_swap_motifs=True,
            num_swaps=2,
        )
        motif_site_cache = {}
        first_stats = AdsorbateGenerationStats(slabs_processed=1)
        second_stats = AdsorbateGenerationStats(slabs_processed=1)
        first_structures = generate_adsorbate_structures_for_slab(
            cfg=supported_cfg,
            atoms=first_heterodimer,
            slab=AseAtomsAdaptor().get_structure(first_heterodimer),
            adsorbate=self.adsorbate,
            atoms_per_layer=self.atoms_per_layer,
            comparison_indices=self.comparison_indices,
            matcher=self.matcher,
            motif_site_cache=motif_site_cache,
            stats=first_stats,
        )
        second_structures = generate_adsorbate_structures_for_slab(
            cfg=supported_cfg,
            atoms=second_heterodimer,
            slab=AseAtomsAdaptor().get_structure(second_heterodimer),
            adsorbate=self.adsorbate,
            atoms_per_layer=self.atoms_per_layer,
            comparison_indices=self.comparison_indices,
            matcher=self.matcher,
            motif_site_cache=motif_site_cache,
            stats=second_stats,
        )
        expected_second_structures = deduplicate_adsorption_structures(
            structures=apply_adsorption_sites(
                cfg=self.cfg,
                structure=AseAtomsAdaptor().get_structure(second_heterodimer),
                adsorbate=self.adsorbate,
                site_coordinates=discover_adsorption_sites(
                    AseAtomsAdaptor().get_structure(second_heterodimer)
                ),
            ),
            comparison_indices=self.comparison_indices,
            matcher=self.matcher,
        )

        self.assertEqual(first_stats.site_finder_calls, 1)
        self.assertGreater(first_stats.matching_seconds, 0.0)
        self.assertEqual(second_stats.site_finder_calls, 1)
        self.assertEqual(second_stats.matching_seconds, 0.0)
        self.assertEqual(len(second_structures), len(expected_second_structures))
        self.assertEqual(
            self._canonical_structure_signatures(second_structures),
            self._canonical_structure_signatures(expected_second_structures),
        )
        self.assertGreater(len(first_structures), 0)

    def test_template_hit_path_does_not_use_matcher_dedup(self):
        first_heterodimer = swap_atoms(self.atoms, 0, "Cu")
        first_heterodimer = swap_atoms(first_heterodimer, 1, "Au")
        second_heterodimer = swap_atoms(self.atoms, 3, "Cu")
        second_heterodimer = swap_atoms(second_heterodimer, 4, "Au")
        supported_cfg = make_config(
            surface_layers_for_matching=2,
            reuse_site_templates_for_two_swap_motifs=True,
            num_swaps=2,
        )
        motif_site_cache = {}
        generate_adsorbate_structures_for_slab(
            cfg=supported_cfg,
            atoms=first_heterodimer,
            slab=AseAtomsAdaptor().get_structure(first_heterodimer),
            adsorbate=self.adsorbate,
            atoms_per_layer=self.atoms_per_layer,
            comparison_indices=self.comparison_indices,
            matcher=self.matcher,
            motif_site_cache=motif_site_cache,
            stats=AdsorbateGenerationStats(slabs_processed=1),
        )

        class FailOnFitMatcher:
            def fit(self, left_structure, right_structure):
                raise AssertionError("matcher dedup should not run on template hits")

        cache_hit_stats = AdsorbateGenerationStats(slabs_processed=1)
        structures = generate_adsorbate_structures_for_slab(
            cfg=supported_cfg,
            atoms=second_heterodimer,
            slab=AseAtomsAdaptor().get_structure(second_heterodimer),
            adsorbate=self.adsorbate,
            atoms_per_layer=self.atoms_per_layer,
            comparison_indices=self.comparison_indices,
            matcher=FailOnFitMatcher(),
            motif_site_cache=motif_site_cache,
            stats=cache_hit_stats,
        )

        self.assertGreater(len(structures), 0)
        self.assertEqual(cache_hit_stats.matching_seconds, 0.0)

    def test_template_hit_path_collapses_overlapping_raw_sites(self):
        first_heterodimer = swap_atoms(self.atoms, 0, "Cu")
        first_heterodimer = swap_atoms(first_heterodimer, 1, "Au")
        second_heterodimer = swap_atoms(self.atoms, 3, "Cu")
        second_heterodimer = swap_atoms(second_heterodimer, 4, "Au")
        supported_cfg = make_config(
            surface_layers_for_matching=2,
            reuse_site_templates_for_two_swap_motifs=True,
            num_swaps=2,
        )
        first_structure = AseAtomsAdaptor().get_structure(first_heterodimer)
        second_structure = AseAtomsAdaptor().get_structure(second_heterodimer)
        motif_site_cache = {}

        generate_adsorbate_structures_for_slab(
            cfg=supported_cfg,
            atoms=first_heterodimer,
            slab=first_structure,
            adsorbate=self.adsorbate,
            atoms_per_layer=self.atoms_per_layer,
            comparison_indices=self.comparison_indices,
            matcher=self.matcher,
            motif_site_cache=motif_site_cache,
            stats=AdsorbateGenerationStats(slabs_processed=1),
        )
        transferred_sites = transfer_adsorption_site_template(
            structure=second_structure,
            template=next(iter(motif_site_cache.values())),
            atoms_per_layer=self.atoms_per_layer,
        )
        overlapping_sites = {
            "hollow": [
                transferred_sites["hollow"][0] + np.array([0.02, 0.0, 0.0]),
                transferred_sites["hollow"][0] + np.array([0.08, 0.0, 0.0]),
                transferred_sites["hollow"][1] + np.array([0.03, 0.0, 0.0]),
            ]
        }
        expected_selected_sites = {
            "hollow": [
                overlapping_sites["hollow"][0],
                overlapping_sites["hollow"][2],
            ]
        }
        expected_structures = apply_adsorption_sites(
            cfg=self.cfg,
            structure=second_structure,
            adsorbate=self.adsorbate,
            site_coordinates=expected_selected_sites,
        )

        with patch(
            "intergen.adsorbate.discover_adsorption_sites",
            return_value=overlapping_sites,
        ):
            generated_structures = generate_adsorbate_structures_for_slab(
                cfg=supported_cfg,
                atoms=second_heterodimer,
                slab=second_structure,
                adsorbate=self.adsorbate,
                atoms_per_layer=self.atoms_per_layer,
                comparison_indices=self.comparison_indices,
                matcher=self.matcher,
                motif_site_cache=motif_site_cache,
                stats=AdsorbateGenerationStats(slabs_processed=1),
            )

        self.assertEqual(len(generated_structures), 2)
        self.assertEqual(
            self._canonical_structure_signatures(generated_structures),
            self._canonical_structure_signatures(expected_structures),
        )

    def test_template_hit_path_uses_configured_tolerance_boundary(self):
        first_heterodimer = swap_atoms(self.atoms, 0, "Cu")
        first_heterodimer = swap_atoms(first_heterodimer, 1, "Au")
        second_heterodimer = swap_atoms(self.atoms, 3, "Cu")
        second_heterodimer = swap_atoms(second_heterodimer, 4, "Au")
        first_structure = AseAtomsAdaptor().get_structure(first_heterodimer)
        second_structure = AseAtomsAdaptor().get_structure(second_heterodimer)
        loose_cfg = make_config(
            surface_layers_for_matching=2,
            reuse_site_templates_for_two_swap_motifs=True,
            num_swaps=2,
            template_site_match_tolerance=0.2,
        )
        strict_cfg = make_config(
            surface_layers_for_matching=2,
            reuse_site_templates_for_two_swap_motifs=True,
            num_swaps=2,
            template_site_match_tolerance=0.19,
        )
        loose_cache = {}
        strict_cache = {}

        generate_adsorbate_structures_for_slab(
            cfg=loose_cfg,
            atoms=first_heterodimer,
            slab=first_structure,
            adsorbate=self.adsorbate,
            atoms_per_layer=self.atoms_per_layer,
            comparison_indices=self.comparison_indices,
            matcher=self.matcher,
            motif_site_cache=loose_cache,
            stats=AdsorbateGenerationStats(slabs_processed=1),
        )
        generate_adsorbate_structures_for_slab(
            cfg=strict_cfg,
            atoms=first_heterodimer,
            slab=first_structure,
            adsorbate=self.adsorbate,
            atoms_per_layer=self.atoms_per_layer,
            comparison_indices=self.comparison_indices,
            matcher=self.matcher,
            motif_site_cache=strict_cache,
            stats=AdsorbateGenerationStats(slabs_processed=1),
        )
        transferred_sites = transfer_adsorption_site_template(
            structure=second_structure,
            template=next(iter(loose_cache.values())),
            atoms_per_layer=self.atoms_per_layer,
        )
        boundary_sites = {
            "hollow": [transferred_sites["hollow"][0] + np.array([0.2, 0.0, 0.0])]
        }

        with patch(
            "intergen.adsorbate.discover_adsorption_sites",
            return_value=boundary_sites,
        ):
            loose_structures = generate_adsorbate_structures_for_slab(
                cfg=loose_cfg,
                atoms=second_heterodimer,
                slab=second_structure,
                adsorbate=self.adsorbate,
                atoms_per_layer=self.atoms_per_layer,
                comparison_indices=self.comparison_indices,
                matcher=self.matcher,
                motif_site_cache=loose_cache,
                stats=AdsorbateGenerationStats(slabs_processed=1),
            )
        with patch(
            "intergen.adsorbate.discover_adsorption_sites",
            return_value=boundary_sites,
        ):
            strict_structures = generate_adsorbate_structures_for_slab(
                cfg=strict_cfg,
                atoms=second_heterodimer,
                slab=second_structure,
                adsorbate=self.adsorbate,
                atoms_per_layer=self.atoms_per_layer,
                comparison_indices=self.comparison_indices,
                matcher=self.matcher,
                motif_site_cache=strict_cache,
                stats=AdsorbateGenerationStats(slabs_processed=1),
            )

        self.assertEqual(len(loose_structures), 1)
        self.assertEqual(len(strict_structures), 0)

    def test_site_template_reuse_can_be_enabled_or_disabled(self):
        first_heterodimer = swap_atoms(self.atoms, 0, "Cu")
        first_heterodimer = swap_atoms(first_heterodimer, 1, "Au")
        second_heterodimer = swap_atoms(self.atoms, 3, "Cu")
        second_heterodimer = swap_atoms(second_heterodimer, 4, "Au")
        enabled_cfg = make_config(
            surface_layers_for_matching=2,
            reuse_site_templates_for_two_swap_motifs=True,
            num_swaps=2,
        )
        disabled_cfg = make_config(
            surface_layers_for_matching=2,
            reuse_site_templates_for_two_swap_motifs=False,
            num_swaps=2,
        )

        enabled_structures, enabled_stats, _ = self._generate_structures_for_atoms_list(
            cfg=enabled_cfg,
            atoms_list=[first_heterodimer, second_heterodimer],
        )
        disabled_structures, disabled_stats, _ = self._generate_structures_for_atoms_list(
            cfg=disabled_cfg,
            atoms_list=[first_heterodimer, second_heterodimer],
        )

        self.assertEqual(enabled_stats.site_finder_calls, 2)
        self.assertEqual(disabled_stats.site_finder_calls, 2)
        self.assertLess(enabled_stats.matching_seconds, disabled_stats.matching_seconds)
        self.assertEqual(
            self._canonical_structure_signatures(enabled_structures),
            self._canonical_structure_signatures(disabled_structures),
        )

    def test_transferred_heterodimer_sites_match_fresh_discovery(self):
        first_heterodimer = swap_atoms(self.atoms, 0, "Cu")
        first_heterodimer = swap_atoms(first_heterodimer, 1, "Au")
        second_heterodimer = swap_atoms(self.atoms, 3, "Cu")
        second_heterodimer = swap_atoms(second_heterodimer, 4, "Au")
        self._assert_transferred_path_matches_direct_path(
            reference_atoms=first_heterodimer,
            target_atoms=second_heterodimer,
            site_name="hollow",
        )

    def test_transferred_dual_single_atom_alloy_sites_match_fresh_discovery(self):
        first_dual_saa = swap_atoms(self.atoms, 0, "Cu")
        first_dual_saa = swap_atoms(first_dual_saa, 4, "Au")
        second_dual_saa = swap_atoms(self.atoms, 2, "Cu")
        second_dual_saa = swap_atoms(second_dual_saa, 7, "Au")
        self._assert_transferred_path_matches_direct_path(
            reference_atoms=first_dual_saa,
            target_atoms=second_dual_saa,
            site_name="hollow",
        )

    def test_pd_heterodimer_fast_path_reuses_cached_site_discovery(self):
        pd_atoms = self._make_host_atoms("Pd")
        first_heterodimer = swap_atoms(pd_atoms, 0, "Cu")
        first_heterodimer = swap_atoms(first_heterodimer, 1, "Au")
        second_heterodimer = swap_atoms(pd_atoms, 3, "Cu")
        second_heterodimer = swap_atoms(second_heterodimer, 4, "Au")
        supported_cfg = make_config(
            surface_layers_for_matching=2,
            reuse_site_templates_for_two_swap_motifs=True,
            num_swaps=2,
        )
        supported_structures, supported_stats, _ = self._generate_structures_for_atoms_list(
            cfg=supported_cfg,
            atoms_list=[first_heterodimer, second_heterodimer],
        )
        exact_structures, exact_stats, _ = self._generate_structures_for_atoms_list(
            cfg=make_config(
                surface_layers_for_matching=2,
                reuse_site_templates_for_two_swap_motifs=False,
                num_swaps=2,
            ),
            atoms_list=[first_heterodimer, second_heterodimer],
        )

        self.assertEqual(supported_stats.site_finder_calls, 2)
        self.assertEqual(exact_stats.site_finder_calls, 2)
        self.assertLess(supported_stats.matching_seconds, exact_stats.matching_seconds)
        self.assertEqual(
            self._canonical_structure_signatures(supported_structures),
            self._canonical_structure_signatures(exact_structures),
        )

    def test_mixed_pt_pd_hosts_do_not_share_cached_templates(self):
        pt_first_heterodimer = swap_atoms(self.atoms, 0, "Cu")
        pt_first_heterodimer = swap_atoms(pt_first_heterodimer, 1, "Au")
        pt_second_heterodimer = swap_atoms(self.atoms, 3, "Cu")
        pt_second_heterodimer = swap_atoms(pt_second_heterodimer, 4, "Au")
        pd_atoms = self._make_host_atoms("Pd")
        pd_first_heterodimer = swap_atoms(pd_atoms, 0, "Cu")
        pd_first_heterodimer = swap_atoms(pd_first_heterodimer, 1, "Au")
        pd_second_heterodimer = swap_atoms(pd_atoms, 3, "Cu")
        pd_second_heterodimer = swap_atoms(pd_second_heterodimer, 4, "Au")
        supported_cfg = make_config(
            surface_layers_for_matching=2,
            reuse_site_templates_for_two_swap_motifs=True,
            num_swaps=2,
        )
        motif_site_cache = {}
        pt_stats = AdsorbateGenerationStats(slabs_processed=1)
        pd_stats = AdsorbateGenerationStats(slabs_processed=1)

        generate_adsorbate_structures_for_slab(
            cfg=supported_cfg,
            atoms=pt_first_heterodimer,
            slab=AseAtomsAdaptor().get_structure(pt_first_heterodimer),
            adsorbate=self.adsorbate,
            atoms_per_layer=self.atoms_per_layer,
            comparison_indices=self.comparison_indices,
            matcher=self.matcher,
            motif_site_cache=motif_site_cache,
            stats=pt_stats,
        )
        generate_adsorbate_structures_for_slab(
            cfg=supported_cfg,
            atoms=pd_first_heterodimer,
            slab=AseAtomsAdaptor().get_structure(pd_first_heterodimer),
            adsorbate=self.adsorbate,
            atoms_per_layer=self.atoms_per_layer,
            comparison_indices=self.comparison_indices,
            matcher=self.matcher,
            motif_site_cache=motif_site_cache,
            stats=pd_stats,
        )

        self.assertEqual(len(motif_site_cache), 2)
        self.assertEqual(pt_stats.site_finder_calls, 1)
        self.assertEqual(pd_stats.site_finder_calls, 1)
        self.assertGreater(pt_stats.matching_seconds, 0.0)
        self.assertGreater(pd_stats.matching_seconds, 0.0)

    def test_pd_transferred_dual_single_atom_alloy_sites_match_fresh_discovery(self):
        pd_atoms = self._make_host_atoms("Pd")
        first_dual_saa = swap_atoms(pd_atoms, 0, "Cu")
        first_dual_saa = swap_atoms(first_dual_saa, 4, "Au")
        second_dual_saa = swap_atoms(pd_atoms, 4, "Cu")
        second_dual_saa = swap_atoms(second_dual_saa, 8, "Au")

        self._assert_transferred_path_matches_direct_path(
            reference_atoms=first_dual_saa,
            target_atoms=second_dual_saa,
            site_name="hollow",
        )

    def test_unsupported_num_swaps_uses_exact_path_and_preserves_results(self):
        first_heterodimer = swap_atoms(self.atoms, 0, "Cu")
        first_heterodimer = swap_atoms(first_heterodimer, 1, "Au")
        second_heterodimer = swap_atoms(self.atoms, 3, "Cu")
        second_heterodimer = swap_atoms(second_heterodimer, 4, "Au")
        fallback_cfg = make_config(
            surface_layers_for_matching=2,
            reuse_site_templates_for_two_swap_motifs=True,
            num_swaps=1,
        )
        exact_cfg = make_config(
            surface_layers_for_matching=2,
            reuse_site_templates_for_two_swap_motifs=False,
            num_swaps=1,
        )
        fallback_stdout = io.StringIO()
        exact_stdout = io.StringIO()

        with redirect_stdout(fallback_stdout):
            fallback_structures = get_adsorbate_structures(
                cfg=fallback_cfg,
                atoms_list=[first_heterodimer, second_heterodimer],
                adsorbate=self.adsorbate,
                matcher=self.matcher,
            )

        with redirect_stdout(exact_stdout):
            exact_structures = get_adsorbate_structures(
                cfg=exact_cfg,
                atoms_list=[first_heterodimer, second_heterodimer],
                adsorbate=self.adsorbate,
                matcher=self.matcher,
            )

        self.assertIn("site_finder_calls=2", fallback_stdout.getvalue())
        self.assertIn("site_finder_calls=2", exact_stdout.getvalue())
        self._assert_structure_lists_match(
            fallback_structures, exact_structures, matcher=self.matcher
        )

    def test_unsupported_motif_still_uses_matcher_dedup(self):
        unsupported_cfg = make_config(
            surface_layers_for_matching=2,
            reuse_site_templates_for_two_swap_motifs=True,
            num_swaps=1,
        )
        counting_matcher = self.CountingMatcher(self.matcher)
        single_swap = swap_atoms(self.atoms, 0, "Cu")
        stats = AdsorbateGenerationStats(slabs_processed=1)

        structures = generate_adsorbate_structures_for_slab(
            cfg=unsupported_cfg,
            atoms=single_swap,
            slab=AseAtomsAdaptor().get_structure(single_swap),
            adsorbate=self.adsorbate,
            atoms_per_layer=self.atoms_per_layer,
            comparison_indices=self.comparison_indices,
            matcher=counting_matcher,
            motif_site_cache={},
            stats=stats,
        )

        self.assertGreater(len(structures), 0)
        self.assertGreater(counting_matcher.fit_calls, 0)
        self.assertGreater(stats.matching_seconds, 0.0)

    def test_unsupported_surface_size_uses_exact_path_and_preserves_results(self):
        large_atoms = fcc111("Pt", size=(2, 3, 4), vacuum=10.0)[::-1]
        large_atoms.set_tags([0] * len(large_atoms))
        large_adsorbate = Molecule(
            ["N"], [[0.0, 0.0, 0.0]], site_properties={"tags": [0]}
        )
        large_matcher = StructureMatcher(ltol=0.05, stol=0.1, angle_tol=5.0)
        first_heterodimer = swap_atoms(large_atoms, 0, "Cu")
        first_heterodimer = swap_atoms(first_heterodimer, 1, "Au")
        second_heterodimer = swap_atoms(large_atoms, 2, "Cu")
        second_heterodimer = swap_atoms(second_heterodimer, 3, "Au")
        fallback_cfg = make_config(
            surface_layers_for_matching=2,
            reuse_site_templates_for_two_swap_motifs=True,
            num_swaps=2,
            size=(2, 3, 4),
        )
        exact_cfg = make_config(
            surface_layers_for_matching=2,
            reuse_site_templates_for_two_swap_motifs=False,
            num_swaps=2,
            size=(2, 3, 4),
        )
        fallback_stdout = io.StringIO()
        exact_stdout = io.StringIO()

        with redirect_stdout(fallback_stdout):
            fallback_structures = get_adsorbate_structures(
                cfg=fallback_cfg,
                atoms_list=[first_heterodimer, second_heterodimer],
                adsorbate=large_adsorbate,
                matcher=large_matcher,
            )

        with redirect_stdout(exact_stdout):
            exact_structures = get_adsorbate_structures(
                cfg=exact_cfg,
                atoms_list=[first_heterodimer, second_heterodimer],
                adsorbate=large_adsorbate,
                matcher=large_matcher,
            )

        self.assertIn("site_finder_calls=2", fallback_stdout.getvalue())
        self.assertIn("site_finder_calls=2", exact_stdout.getvalue())
        self._assert_structure_lists_match(
            fallback_structures, exact_structures, matcher=large_matcher
        )


class TestTopLayerMotifClassification(unittest.TestCase):
    def setUp(self):
        self.atoms = fcc111("Pt", size=(3, 3, 4), vacuum=10.0)[::-1]
        self.atoms_per_layer = 9

    def test_classifies_pure_top_layer(self):
        motif = classify_top_layer_motif(self.atoms, self.atoms_per_layer)

        self.assertEqual(motif, "pure")

    def test_classifies_single_swap_top_layer(self):
        swapped = swap_atoms(self.atoms, 0, "Cu")

        motif = classify_top_layer_motif(swapped, self.atoms_per_layer)

        self.assertEqual(motif, "single_swap")

    def test_classifies_heterodimer_top_layer(self):
        swapped = swap_atoms(self.atoms, 0, "Cu")
        swapped = swap_atoms(swapped, 1, "Au")

        motif = classify_top_layer_motif(swapped, self.atoms_per_layer)

        self.assertEqual(motif, "heterodimer")

    def test_classifies_dual_single_atom_alloy_top_layer(self):
        swapped = swap_atoms(self.atoms, 0, "Cu")
        swapped = swap_atoms(swapped, 4, "Au")

        motif = classify_top_layer_motif(swapped, self.atoms_per_layer)

        self.assertEqual(motif, "dual_single_atom_alloy")


if __name__ == "__main__":
    unittest.main()
