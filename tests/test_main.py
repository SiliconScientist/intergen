import tempfile
import unittest
from pathlib import Path

from types import SimpleNamespace

from ase import Atoms
from ase.constraints import FixAtoms
from ase.db import connect

from intergen.__main__ import (
    apply_database_constraints,
    get_database_atoms_list,
    get_initial_atoms_list,
    write_atoms_database,
)
from intergen.metadata import (
    ADSLAB_ID_KEY,
    ADSORBATE_KEY,
    DB_METADATA_DATA_KEY,
    HOST_ELEMENT_KEY,
    INITIAL_SITE_COORDINATE_KEY,
    INITIAL_SITE_LABEL_KEY,
    PARENT_SLAB_ID_KEY,
    SUPERCELL_SIZE_KEY,
    SURFACE_TYPE_KEY,
    SWAP_ELEMENTS_KEY,
    SWAP_INDICES_KEY,
    TOP_LAYER_MOTIF_KEY,
)


class TestMain(unittest.TestCase):
    def test_get_initial_atoms_list_includes_pure_atoms_by_default(self):
        pure_atoms = ["slab-a", "slab-b"]

        atoms_list = get_initial_atoms_list(
            pure_atoms=pure_atoms,
            only_last_generation=False,
        )

        self.assertEqual(atoms_list, pure_atoms)
        self.assertIsNot(atoms_list, pure_atoms)

    def test_get_initial_atoms_list_skips_pure_atoms_for_last_generation_only(self):
        pure_atoms = ["slab-a", "slab-b"]

        atoms_list = get_initial_atoms_list(
            pure_atoms=pure_atoms,
            only_last_generation=True,
        )

        self.assertEqual(atoms_list, [])


    def test_apply_database_constraints_is_noop_when_disabled(self):
        atoms = Atoms(
            symbols=['Pt', 'Pt', 'N'],
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.5, 1.2)],
            cell=[(5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 15.0)],
            pbc=(True, True, True),
        )
        cfg = SimpleNamespace(
            database=SimpleNamespace(
                constrain_bottom_layers=0,
                constraint_z_tolerance=0.5,
                constraint_lowest_z_tolerance=0.5,
            ),
            adsorbate=SimpleNamespace(coords=[(0.0, 0.0, 0.0)]),
        )

        constrained = apply_database_constraints(cfg=cfg, atoms_list=[atoms])

        self.assertEqual(len(constrained), 1)
        self.assertEqual(constrained[0].constraints, [])

    def test_apply_database_constraints_adds_fixatoms_when_enabled(self):
        slab_positions = []
        for z in (0.0, 2.0, 4.0):
            for x in (0.0, 1.5):
                for y in (0.0, 1.5):
                    slab_positions.append((x, y, z))
        atoms = Atoms(
            symbols=['Pt'] * len(slab_positions) + ['N'],
            positions=slab_positions + [(0.75, 0.75, 5.5)],
            cell=[(6.0, 0.0, 0.0), (0.0, 6.0, 0.0), (0.0, 0.0, 18.0)],
            pbc=(True, True, True),
        )
        cfg = SimpleNamespace(
            database=SimpleNamespace(
                constrain_bottom_layers=2,
                constraint_z_tolerance=0.5,
                constraint_lowest_z_tolerance=0.5,
            ),
            adsorbate=SimpleNamespace(coords=[(0.0, 0.0, 0.0)]),
        )

        constrained = apply_database_constraints(cfg=cfg, atoms_list=[atoms])[0]

        self.assertEqual(len(constrained.constraints), 1)
        self.assertIsInstance(constrained.constraints[0], FixAtoms)
        self.assertEqual(sorted(constrained.constraints[0].index.tolist()), list(range(8)))

    def test_write_atoms_database_persists_metadata_in_db_row(self):
        atoms = Atoms(
            symbols=["Pt", "N"],
            positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
            cell=[(5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 15.0)],
            pbc=(True, True, True),
        )
        atoms.info.update(
            {
                ADSLAB_ID_KEY: "adslab-000001",
                PARENT_SLAB_ID_KEY: "slab-000010",
                HOST_ELEMENT_KEY: "Pt",
                SURFACE_TYPE_KEY: "fcc111",
                SUPERCELL_SIZE_KEY: (3, 3, 4),
                SWAP_INDICES_KEY: [0, 4],
                SWAP_ELEMENTS_KEY: ["Cu", "Au"],
                TOP_LAYER_MOTIF_KEY: "heterodimer",
                ADSORBATE_KEY: "N",
                INITIAL_SITE_LABEL_KEY: "fcc_hollow",
                INITIAL_SITE_COORDINATE_KEY: (1.0, 2.0, 3.0),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "metadata.db"
            write_atoms_database(db_path, [atoms])

            row = next(connect(db_path).select())

        self.assertEqual(row.adslab_id, "adslab-000001")
        self.assertEqual(row.parent_slab_id, "slab-000010")
        self.assertEqual(row.host_element, "Pt")
        self.assertEqual(row.surface_type, "fcc111")
        self.assertEqual(row.top_layer_motif, "heterodimer")
        self.assertEqual(row.adsorbate, "N")
        self.assertEqual(row.initial_site_label, "fcc_hollow")
        self.assertEqual(row.key_value_pairs[SUPERCELL_SIZE_KEY], "[3, 3, 4]")
        self.assertEqual(row.key_value_pairs[SWAP_INDICES_KEY], "[0, 4]")
        self.assertEqual(row.key_value_pairs[SWAP_ELEMENTS_KEY], '["Cu", "Au"]')
        self.assertEqual(row.key_value_pairs[INITIAL_SITE_COORDINATE_KEY], "[1.0, 2.0, 3.0]")
        self.assertEqual(
            tuple(row.data[DB_METADATA_DATA_KEY][INITIAL_SITE_COORDINATE_KEY]),
            (1.0, 2.0, 3.0),
        )

    def test_get_database_atoms_list_restores_metadata_round_trip(self):
        atoms = Atoms(
            symbols=["Pt", "N"],
            positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
            cell=[(5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 15.0)],
            pbc=(True, True, True),
        )
        atoms.info.update(
            {
                ADSLAB_ID_KEY: "adslab-000001",
                PARENT_SLAB_ID_KEY: "slab-000010",
                HOST_ELEMENT_KEY: "Pt",
                SURFACE_TYPE_KEY: "fcc111",
                SUPERCELL_SIZE_KEY: (3, 3, 4),
                SWAP_INDICES_KEY: [0, 4],
                SWAP_ELEMENTS_KEY: ["Cu", "Au"],
                TOP_LAYER_MOTIF_KEY: "heterodimer",
                ADSORBATE_KEY: "N",
                INITIAL_SITE_LABEL_KEY: "fcc_hollow",
                INITIAL_SITE_COORDINATE_KEY: (1.0, 2.0, 3.0),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "metadata.db"
            write_atoms_database(db_path, [atoms])
            restored_atoms = get_database_atoms_list(db_path)

        self.assertEqual(len(restored_atoms), 1)
        self.assertEqual(restored_atoms[0].info[ADSLAB_ID_KEY], "adslab-000001")
        self.assertEqual(restored_atoms[0].info[PARENT_SLAB_ID_KEY], "slab-000010")
        self.assertEqual(restored_atoms[0].info[HOST_ELEMENT_KEY], "Pt")
        self.assertEqual(restored_atoms[0].info[SURFACE_TYPE_KEY], "fcc111")
        self.assertEqual(restored_atoms[0].info[SUPERCELL_SIZE_KEY], (3, 3, 4))
        self.assertEqual(restored_atoms[0].info[SWAP_INDICES_KEY], [0, 4])
        self.assertEqual(restored_atoms[0].info[SWAP_ELEMENTS_KEY], ["Cu", "Au"])
        self.assertEqual(restored_atoms[0].info[TOP_LAYER_MOTIF_KEY], "heterodimer")
        self.assertEqual(restored_atoms[0].info[ADSORBATE_KEY], "N")
        self.assertEqual(restored_atoms[0].info[INITIAL_SITE_LABEL_KEY], "fcc_hollow")
        self.assertEqual(
            restored_atoms[0].info[INITIAL_SITE_COORDINATE_KEY],
            (1.0, 2.0, 3.0),
        )


if __name__ == "__main__":
    unittest.main()
