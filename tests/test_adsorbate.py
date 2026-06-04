import io
import unittest
from contextlib import redirect_stdout

import numpy as np
from ase.build import fcc111
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Molecule
from pymatgen.io.ase import AseAtomsAdaptor

from intergen.adsorbate import (
    build_adsorption_site_template,
    add_adsorbates,
    apply_adsorption_sites,
    build_adsorbate_comparison_substructures,
    deduplicate_adsorption_structures,
    discover_adsorption_sites,
    generate_adsorbate_structures_for_slab,
    get_adsorbate_comparison_indices,
    get_adsorbate_structures,
    get_top_layer_host_element,
    resolve_adsorption_sites,
    supports_two_swap_motif_template_reuse,
    transfer_adsorption_site_template,
)
from intergen.config import Config
from intergen.surface import (
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
):
    return Config(
        structure={
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


class TestHollowSiteRegistry(unittest.TestCase):
    def setUp(self):
        self.cfg = make_config(surface_layers_for_matching=2)
        atoms = self._make_host_atoms("Pt")
        self.atoms = atoms
        atoms.set_pbc(True)
        self.slab = AseAtomsAdaptor().get_structure(atoms)
        self.site_finder = AdsorbateSiteFinder(self.slab)
        self.adsorbate = Molecule(
            ["N"], [[0.0, 0.0, 0.0]], site_properties={"tags": [0]}
        )
        self.matcher = StructureMatcher(ltol=0.05, stol=0.1, angle_tol=5.0)
        self.atoms_per_layer = 9
        self.adsorbate_index = [len(self.slab)]

    def _make_host_atoms(self, host, size=(3, 3, 4)):
        atoms = fcc111(host, size=size, vacuum=10.0)[::-1]
        atoms.set_tags([0] * len(atoms))
        atoms.set_pbc(True)
        return atoms

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

    def _get_unique_adsorbate_structures(self, structures):
        representative_indices = find_unique_structures(
            self._comparison_substructures(
                structures,
                surface_layers=self.cfg.adsorbate.surface_layers_for_matching,
            ),
            matcher=self.matcher,
        )
        return [structures[index] for index in representative_indices]

    def _assert_structure_lists_match(
        self, left_structures, right_structures, matcher=None
    ):
        if matcher is None:
            matcher = self.matcher
        converter = AseAtomsAdaptor()
        left_structures = [converter.get_structure(atoms) for atoms in left_structures]
        right_structures = [converter.get_structure(atoms) for atoms in right_structures]
        self.assertEqual(len(left_structures), len(right_structures))
        unmatched_indices = list(range(len(right_structures)))
        for left_structure in left_structures:
            for index in unmatched_indices:
                if matcher.fit(left_structure, right_structures[index]):
                    unmatched_indices.remove(index)
                    break
            else:
                self.fail("Could not match adsorbate structure between paths.")
        self.assertEqual(unmatched_indices, [])

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
            len(unique_transferred_structures),
            len(unique_direct_structures),
        )
        for transferred_structure, direct_structure in zip(
            unique_transferred_structures,
            unique_direct_structures,
        ):
            self.assertTrue(self.matcher.fit(transferred_structure, direct_structure))

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
        hollow_sites = self.site_finder.find_adsorption_sites(symm_reduce=False)[
            "hollow"
        ]
        hollow_structures = [
            self.site_finder.add_adsorbate(self.adsorbate, coords)
            for coords in hollow_sites
        ]

        two_layer_substructures = self._comparison_substructures(
            hollow_structures, surface_layers=2
        )
        representative_indices = find_unique_structures(
            two_layer_substructures, matcher=self.matcher
        )
        representative_structures = [
            hollow_structures[index] for index in representative_indices
        ]

        self.assertEqual(len(representative_structures), 2)

        one_layer_unique = find_unique_structures(
            self._comparison_substructures(
                representative_structures, surface_layers=1
            ),
            matcher=self.matcher,
        )
        two_layer_unique = find_unique_structures(
            self._comparison_substructures(
                representative_structures, surface_layers=2
            ),
            matcher=self.matcher,
        )

        self.assertEqual(one_layer_unique, [0])
        self.assertEqual(two_layer_unique, [0, 1])

    def test_get_adsorbate_structures_distinguishes_hollow_registry_by_layer_count(self):
        one_layer_cfg = make_config(surface_layers_for_matching=1)
        two_layer_cfg = make_config(surface_layers_for_matching=2)

        one_layer_structures = get_adsorbate_structures(
            cfg=one_layer_cfg,
            atoms_list=[self.atoms],
            adsorbate=self.adsorbate,
            matcher=self.matcher,
        )
        two_layer_structures = get_adsorbate_structures(
            cfg=two_layer_cfg,
            atoms_list=[self.atoms],
            adsorbate=self.adsorbate,
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
        discovered_sites = discover_adsorption_sites(self.slab)

        split_structures = apply_adsorption_sites(
            cfg=self.cfg,
            structure=self.slab,
            adsorbate=self.adsorbate,
            site_coordinates=discovered_sites,
        )
        direct_structures = add_adsorbates(
            cfg=self.cfg,
            structure=self.slab,
            adsorbate=self.adsorbate,
        )

        self.assertEqual(len(split_structures), len(direct_structures))

        for split_structure, direct_structure in zip(split_structures, direct_structures):
            self.assertTrue(self.matcher.fit(split_structure, direct_structure))

    def test_build_adsorbate_comparison_substructures_matches_existing_helper(self):
        discovered_sites = discover_adsorption_sites(self.slab)
        structures = apply_adsorption_sites(
            cfg=self.cfg,
            structure=self.slab,
            adsorbate=self.adsorbate,
            site_coordinates=discovered_sites,
        )
        comparison_indices = get_adsorbate_comparison_indices(
            atoms_per_layer=self.atoms_per_layer,
            surface_layers=self.cfg.adsorbate.surface_layers_for_matching,
            adsorbate_indices=self.adsorbate_index,
        )

        substructures = build_adsorbate_comparison_substructures(
            structures=structures,
            comparison_indices=comparison_indices,
        )
        expected_substructures = self._comparison_substructures(
            structures,
            surface_layers=self.cfg.adsorbate.surface_layers_for_matching,
        )

        self.assertEqual(len(substructures), len(expected_substructures))
        for substructure, expected_substructure in zip(
            substructures, expected_substructures
        ):
            self.assertTrue(self.matcher.fit(substructure, expected_substructure))

    def test_deduplicate_adsorption_structures_matches_existing_helper(self):
        discovered_sites = discover_adsorption_sites(self.slab)
        structures = apply_adsorption_sites(
            cfg=self.cfg,
            structure=self.slab,
            adsorbate=self.adsorbate,
            site_coordinates=discovered_sites,
        )
        comparison_indices = get_adsorbate_comparison_indices(
            atoms_per_layer=self.atoms_per_layer,
            surface_layers=self.cfg.adsorbate.surface_layers_for_matching,
            adsorbate_indices=self.adsorbate_index,
        )

        deduplicated_structures = deduplicate_adsorption_structures(
            structures=structures,
            comparison_indices=comparison_indices,
            matcher=self.matcher,
        )
        expected_structures = self._get_unique_adsorbate_structures(structures)

        self.assertEqual(len(deduplicated_structures), len(expected_structures))
        for deduplicated_structure, expected_structure in zip(
            deduplicated_structures, expected_structures
        ):
            self.assertTrue(self.matcher.fit(deduplicated_structure, expected_structure))

    def test_generate_adsorbate_structures_for_slab_matches_split_workflow(self):
        motif_site_cache = {}
        comparison_indices = get_adsorbate_comparison_indices(
            atoms_per_layer=self.atoms_per_layer,
            surface_layers=self.cfg.adsorbate.surface_layers_for_matching,
            adsorbate_indices=self.adsorbate_index,
        )

        site_coordinates = resolve_adsorption_sites(
            cfg=self.cfg,
            atoms=self.atoms,
            structure=self.slab,
            atoms_per_layer=self.atoms_per_layer,
            motif_site_cache=motif_site_cache,
        )
        structures = apply_adsorption_sites(
            cfg=self.cfg,
            structure=self.slab,
            adsorbate=self.adsorbate,
            site_coordinates=site_coordinates,
        )
        expected_structures = deduplicate_adsorption_structures(
            structures=structures,
            comparison_indices=comparison_indices,
            matcher=self.matcher,
        )

        slab_structures = generate_adsorbate_structures_for_slab(
            cfg=self.cfg,
            atoms=self.atoms,
            slab=self.slab,
            adsorbate=self.adsorbate,
            atoms_per_layer=self.atoms_per_layer,
            comparison_indices=comparison_indices,
            matcher=self.matcher,
            motif_site_cache={},
        )

        self.assertEqual(len(slab_structures), len(expected_structures))
        for slab_structure, expected_structure in zip(
            slab_structures, expected_structures
        ):
            self.assertTrue(self.matcher.fit(slab_structure, expected_structure))

    def test_equivalent_heterodimer_slabs_reuse_cached_site_discovery(self):
        first_heterodimer = swap_atoms(self.atoms, 0, "Cu")
        first_heterodimer = swap_atoms(first_heterodimer, 1, "Au")
        second_heterodimer = swap_atoms(self.atoms, 3, "Cu")
        second_heterodimer = swap_atoms(second_heterodimer, 4, "Au")
        stdout = io.StringIO()
        supported_cfg = make_config(
            surface_layers_for_matching=2,
            reuse_site_templates_for_two_swap_motifs=True,
            num_swaps=2,
        )

        with redirect_stdout(stdout):
            structures = get_adsorbate_structures(
                cfg=supported_cfg,
                atoms_list=[first_heterodimer, second_heterodimer],
                adsorbate=self.adsorbate,
                matcher=self.matcher,
            )

        output = stdout.getvalue()

        self.assertGreater(len(structures), 0)
        self.assertIn("slabs=2", output)
        self.assertIn("site_finder_calls=1", output)

    def test_site_template_reuse_can_be_enabled_or_disabled(self):
        first_heterodimer = swap_atoms(self.atoms, 0, "Cu")
        first_heterodimer = swap_atoms(first_heterodimer, 1, "Au")
        second_heterodimer = swap_atoms(self.atoms, 3, "Cu")
        second_heterodimer = swap_atoms(second_heterodimer, 4, "Au")
        enabled_stdout = io.StringIO()
        disabled_stdout = io.StringIO()
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

        with redirect_stdout(enabled_stdout):
            get_adsorbate_structures(
                cfg=enabled_cfg,
                atoms_list=[first_heterodimer, second_heterodimer],
                adsorbate=self.adsorbate,
                matcher=self.matcher,
            )

        with redirect_stdout(disabled_stdout):
            get_adsorbate_structures(
                cfg=disabled_cfg,
                atoms_list=[first_heterodimer, second_heterodimer],
                adsorbate=self.adsorbate,
                matcher=self.matcher,
            )

        self.assertIn("site_finder_calls=1", enabled_stdout.getvalue())
        self.assertIn("site_finder_calls=2", disabled_stdout.getvalue())

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
        stdout = io.StringIO()
        supported_cfg = make_config(
            surface_layers_for_matching=2,
            reuse_site_templates_for_two_swap_motifs=True,
            num_swaps=2,
        )

        with redirect_stdout(stdout):
            structures = get_adsorbate_structures(
                cfg=supported_cfg,
                atoms_list=[first_heterodimer, second_heterodimer],
                adsorbate=self.adsorbate,
                matcher=self.matcher,
            )

        output = stdout.getvalue()

        self.assertGreater(len(structures), 0)
        self.assertIn("slabs=2", output)
        self.assertIn("site_finder_calls=1", output)

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
        stdout = io.StringIO()
        supported_cfg = make_config(
            surface_layers_for_matching=2,
            reuse_site_templates_for_two_swap_motifs=True,
            num_swaps=2,
        )

        with redirect_stdout(stdout):
            structures = get_adsorbate_structures(
                cfg=supported_cfg,
                atoms_list=[
                    pt_first_heterodimer,
                    pt_second_heterodimer,
                    pd_first_heterodimer,
                    pd_second_heterodimer,
                ],
                adsorbate=self.adsorbate,
                matcher=self.matcher,
            )

        output = stdout.getvalue()

        self.assertGreater(len(structures), 0)
        self.assertIn("slabs=4", output)
        self.assertIn("site_finder_calls=2", output)

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
