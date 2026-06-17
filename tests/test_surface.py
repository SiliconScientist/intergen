import unittest
from itertools import count
from unittest.mock import patch

from ase.build import bcc111, fcc111

from intergen.config import Config
from intergen.metadata import (
    HOST_ELEMENT_KEY,
    SLAB_ID_KEY,
    SUPERCELL_SIZE_KEY,
    SURFACE_TYPE_BCC111,
    SURFACE_TYPE_FCC111,
    SURFACE_TYPE_HCP0001,
    SURFACE_TYPE_KEY,
    SWAP_ELEMENTS_KEY,
    SWAP_INDICES_KEY,
    TOP_LAYER_MOTIF_KEY,
    TOP_LAYER_MOTIF_DUAL_SINGLE_ATOM_ALLOY,
    TOP_LAYER_MOTIF_HETERODIMER,
    TOP_LAYER_MOTIF_PURE,
    TOP_LAYER_MOTIF_SINGLE_SWAP,
)
from intergen.surface import (
    assign_slab_metadata,
    build_pure_surfaces,
    classify_top_layer_motif,
    enumerate_unique_swaps,
    iterative_swaps,
    swap_atoms,
)


def make_config():
    return Config(
        structure={
            "bcc": {"elements": ["Fe"], "size": (2, 2, 3)},
            "hcp": {"elements": ["Co"], "size": (2, 2, 3)},
            "fcc": {"elements": ["Cu"], "size": (2, 2, 3)},
            "vacuum": 10.0,
        },
        generation={
            "layers_to_swap": 1,
            "num_swaps": 2,
            "swap_elements": ["Ni", "Zn"],
            "only_last_generation": False,
        },
        database={"path": "data/test.db"},
        adsorbate={
            "matcher": {"ltol": 0.05, "stol": 0.1, "angle_tol": 5.0},
            "species": "N",
            "coords": [(0.0, 0.0, 0.0)],
            "sites": ["hollow"],
            "tag": 0,
            "surface_layers_for_matching": 2,
        },
    )


class TestSurfaceMetadata(unittest.TestCase):
    def test_build_pure_surfaces_assigns_slab_metadata(self):
        cfg = make_config()

        pure_atoms = build_pure_surfaces(cfg=cfg, slab_id_source=count(1))

        self.assertEqual(len(pure_atoms), 3)

        fcc_slab = pure_atoms[0]
        self.assertEqual(fcc_slab.info[SLAB_ID_KEY], "slab-000001")
        self.assertEqual(fcc_slab.info[HOST_ELEMENT_KEY], "Cu")
        self.assertEqual(fcc_slab.info[SURFACE_TYPE_KEY], SURFACE_TYPE_FCC111)
        self.assertEqual(fcc_slab.info[SUPERCELL_SIZE_KEY], (2, 2, 3))
        self.assertEqual(fcc_slab.info[SWAP_INDICES_KEY], [])
        self.assertEqual(fcc_slab.info[SWAP_ELEMENTS_KEY], [])
        self.assertEqual(fcc_slab.info[TOP_LAYER_MOTIF_KEY], TOP_LAYER_MOTIF_PURE)

        bcc_slab = pure_atoms[1]
        self.assertEqual(bcc_slab.info[SLAB_ID_KEY], "slab-000002")
        self.assertEqual(bcc_slab.info[HOST_ELEMENT_KEY], "Fe")
        self.assertEqual(bcc_slab.info[SURFACE_TYPE_KEY], SURFACE_TYPE_BCC111)
        self.assertEqual(bcc_slab.info[SUPERCELL_SIZE_KEY], (2, 2, 3))
        self.assertEqual(bcc_slab.info[SWAP_INDICES_KEY], [])
        self.assertEqual(bcc_slab.info[SWAP_ELEMENTS_KEY], [])
        self.assertEqual(bcc_slab.info[TOP_LAYER_MOTIF_KEY], TOP_LAYER_MOTIF_PURE)

        hcp_slab = pure_atoms[2]
        self.assertEqual(hcp_slab.info[SLAB_ID_KEY], "slab-000003")
        self.assertEqual(hcp_slab.info[HOST_ELEMENT_KEY], "Co")
        self.assertEqual(hcp_slab.info[SURFACE_TYPE_KEY], SURFACE_TYPE_HCP0001)
        self.assertEqual(hcp_slab.info[SUPERCELL_SIZE_KEY], (2, 2, 3))
        self.assertEqual(hcp_slab.info[SWAP_INDICES_KEY], [])
        self.assertEqual(hcp_slab.info[SWAP_ELEMENTS_KEY], [])
        self.assertEqual(hcp_slab.info[TOP_LAYER_MOTIF_KEY], TOP_LAYER_MOTIF_PURE)

    def test_enumerate_unique_swaps_assigns_new_slab_id_and_swap_provenance(self):
        atoms = fcc111("Pt", size=(2, 2, 3), vacuum=10.0)[::-1]
        assign_slab_metadata(
            atoms,
            slab_id="slab-000010",
            host_element="Pt",
            surface_type=SURFACE_TYPE_FCC111,
            supercell_size=(2, 2, 3),
        )

        with patch("intergen.surface.id_unique_sites", return_value=[0]):
            swapped_atoms = enumerate_unique_swaps(
                atoms=atoms,
                host_element="Pt",
                indices=[0, 1],
                element="Cu",
                slab_id_source=count(11),
            )

        self.assertEqual(len(swapped_atoms), 1)
        swapped = swapped_atoms[0]
        self.assertEqual(swapped.info[SLAB_ID_KEY], "slab-000011")
        self.assertEqual(swapped.info[HOST_ELEMENT_KEY], "Pt")
        self.assertEqual(swapped.info[SURFACE_TYPE_KEY], SURFACE_TYPE_FCC111)
        self.assertEqual(swapped.info[SUPERCELL_SIZE_KEY], (2, 2, 3))
        self.assertEqual(swapped.info[SWAP_INDICES_KEY], [0])
        self.assertEqual(swapped.info[SWAP_ELEMENTS_KEY], ["Cu"])
        self.assertEqual(swapped.info[TOP_LAYER_MOTIF_KEY], TOP_LAYER_MOTIF_SINGLE_SWAP)

        self.assertEqual(atoms.info[SLAB_ID_KEY], "slab-000010")
        self.assertEqual(atoms.info[SWAP_INDICES_KEY], [])
        self.assertEqual(atoms.info[SWAP_ELEMENTS_KEY], [])

    def test_enumerate_unique_swaps_preserves_bcc_surface_type(self):
        atoms = bcc111("Fe", size=(2, 2, 3), vacuum=10.0)[::-1]
        assign_slab_metadata(
            atoms,
            slab_id="slab-000010",
            host_element="Fe",
            surface_type=SURFACE_TYPE_BCC111,
            supercell_size=(2, 2, 3),
        )

        with patch("intergen.surface.id_unique_sites", return_value=[0]):
            swapped_atoms = enumerate_unique_swaps(
                atoms=atoms,
                host_element="Fe",
                indices=[0, 1],
                element="Cu",
                slab_id_source=count(11),
            )

        self.assertEqual(len(swapped_atoms), 1)
        self.assertEqual(swapped_atoms[0].info[SURFACE_TYPE_KEY], SURFACE_TYPE_BCC111)

    def test_iterative_swaps_preserves_and_accumulates_swap_metadata(self):
        cfg = make_config()
        slab_id_source = count(1)
        pure_atoms = build_pure_surfaces(cfg=cfg, slab_id_source=slab_id_source)

        with patch("intergen.surface.id_unique_sites", side_effect=[[0], [1]]):
            swapped_atoms = iterative_swaps(
                atoms=pure_atoms[0],
                host_element="Cu",
                swap_plan=["Ni", "Zn"],
                indices=[0, 1],
                atoms_per_layer=4,
                slab_id_source=slab_id_source,
                only_last_generation=False,
            )

        self.assertEqual(len(swapped_atoms), 2)
        first_generation, second_generation = swapped_atoms

        self.assertEqual(first_generation.info[SLAB_ID_KEY], "slab-000004")
        self.assertEqual(first_generation.info[SWAP_INDICES_KEY], [0])
        self.assertEqual(first_generation.info[SWAP_ELEMENTS_KEY], ["Ni"])
        self.assertEqual(
            first_generation.info[TOP_LAYER_MOTIF_KEY], TOP_LAYER_MOTIF_SINGLE_SWAP
        )

        self.assertEqual(second_generation.info[SLAB_ID_KEY], "slab-000005")
        self.assertEqual(second_generation.info[SWAP_INDICES_KEY], [0, 1])
        self.assertEqual(second_generation.info[SWAP_ELEMENTS_KEY], ["Ni", "Zn"])
        self.assertEqual(
            second_generation.info[TOP_LAYER_MOTIF_KEY],
            classify_top_layer_motif(second_generation, atoms_per_layer=4),
        )

    def test_classify_top_layer_motif_distinguishes_heterodimer(self):
        atoms = fcc111("Pt", size=(3, 3, 3), vacuum=10.0)[::-1]
        atoms = swap_atoms(atoms, 0, "Cu")
        atoms = swap_atoms(atoms, 1, "Au")

        motif = classify_top_layer_motif(atoms, atoms_per_layer=9)

        self.assertEqual(motif, TOP_LAYER_MOTIF_HETERODIMER)

    def test_classify_top_layer_motif_distinguishes_dual_single_atom_alloy(self):
        atoms = fcc111("Pt", size=(3, 3, 3), vacuum=10.0)[::-1]
        atoms = swap_atoms(atoms, 0, "Cu")
        atoms = swap_atoms(atoms, 4, "Au")

        motif = classify_top_layer_motif(atoms, atoms_per_layer=9)

        self.assertEqual(motif, TOP_LAYER_MOTIF_DUAL_SINGLE_ATOM_ALLOY)


if __name__ == "__main__":
    unittest.main()
