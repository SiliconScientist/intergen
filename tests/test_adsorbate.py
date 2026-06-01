import unittest

from ase import Atoms
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Molecule

from intergen.adsorbate import find_unique_adsorbate_structures
from intergen.config import get_config
from intergen.surface import prepare_for_pymatgen


FRAME_6 = Atoms(
    symbols=[
        "Cu", "Pt", "Pt", "Pt", "Pt", "Pt", "Pt", "Pt", "Pt", "Pt", "Pt", "Pt",
        "Pt", "Pt", "Pt", "Pt", "Pt", "Pt", "Pt", "Pt", "Pt", "Pt", "Pt", "Pt",
        "Pt", "Pt", "Pt", "Pt", "Pt", "Pt", "Pt", "Pt", "Pt", "Pt", "Pt", "Pt",
        "N",
    ],
    positions=[
        [8.34109523, 4.81573358, 16.81047574],
        [5.56073015, 4.81573358, 16.81047574],
        [2.78036508, 4.81573358, 16.81047574],
        [6.95091269, 2.40786679, 16.81047574],
        [4.17054762, 2.40786679, 16.81047574],
        [1.39018254, 2.40786679, 16.81047574],
        [5.56073015, 0.0, 16.81047574],
        [2.78036508, 0.0, 16.81047574],
        [0.0, 0.0, 16.81047574],
        [8.34109523, 6.4209781, 14.54031716],
        [5.56073015, 6.4209781, 14.54031716],
        [2.78036508, 6.4209781, 14.54031716],
        [6.95091269, 4.01311131, 14.54031716],
        [4.17054762, 4.01311131, 14.54031716],
        [1.39018254, 4.01311131, 14.54031716],
        [5.56073015, 1.60524453, 14.54031716],
        [2.78036508, 1.60524453, 14.54031716],
        [0.0, 1.60524453, 14.54031716],
        [9.73127777, 5.61835584, 12.27015858],
        [6.95091269, 5.61835584, 12.27015858],
        [4.17054762, 5.61835584, 12.27015858],
        [8.34109523, 3.21048905, 12.27015858],
        [5.56073015, 3.21048905, 12.27015858],
        [2.78036508, 3.21048905, 12.27015858],
        [6.95091269, 0.80262226, 12.27015858],
        [4.17054762, 0.80262226, 12.27015858],
        [1.39018254, 0.80262226, 12.27015858],
        [8.34109523, 4.81573358, 10.0],
        [5.56073015, 4.81573358, 10.0],
        [2.78036508, 4.81573358, 10.0],
        [6.95091269, 2.40786679, 10.0],
        [4.17054762, 2.40786679, 10.0],
        [1.39018254, 2.40786679, 10.0],
        [5.56073015, 0.0, 10.0],
        [2.78036508, 0.0, 10.0],
        [0.0, 0.0, 10.0],
        [5.56073015, 4.81573358, 18.81047574],
    ],
    cell=[
        [8.34109523, 0.0, 0.0],
        [4.17054762, 7.22360036, 0.0],
        [0.0, 0.0, 26.81047574],
    ],
    pbc=[True, True, True],
)

FRAME_7 = Atoms(
    symbols=FRAME_6.get_chemical_symbols(),
    positions=FRAME_6.positions[:-1].tolist() + [[4.17054762, 2.40786679, 18.81047574]],
    cell=FRAME_6.cell,
    pbc=FRAME_6.pbc,
)


class AdsorbateMatchingTests(unittest.TestCase):
    def test_frames_6_and_7_are_treated_as_one_adsorbate_site(self):
        cfg = get_config()
        atoms = [FRAME_6.copy(), FRAME_7.copy()]
        structures = prepare_for_pymatgen(atoms)
        atoms_per_layer = cfg.structure.size[0] * cfg.structure.size[1]
        surface_indices = list(
            range(atoms_per_layer * cfg.adsorbate.surface_layers_for_matching)
        )
        adsorbate = Molecule(
            species=cfg.adsorbate.species,
            coords=cfg.adsorbate.coords,
        )
        matcher = StructureMatcher(**cfg.adsorbate.matcher.model_dump())

        unique_indices = find_unique_adsorbate_structures(
            structures=structures,
            surface_indices=surface_indices,
            adsorbate=adsorbate,
            matcher=matcher,
        )

        self.assertEqual(unique_indices, [0])


if __name__ == "__main__":
    unittest.main()
