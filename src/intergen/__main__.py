from ase.db import connect
from ase.visualize import view
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Molecule
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.analysis.structure_matcher import StructureMatcher
from intergen.config import get_config
from intergen.surface import (
    build_pure_surfaces,
    naive_surface_index_selector,
    mutate_via_swaps,
)
from intergen.adsorbate import get_adsorbate_structures


def main():
    cfg = get_config()
    if cfg.database.path.exists():
        print(f"Output already exists at {cfg.database.path}. Skipping generation.")
        return
    matcher = StructureMatcher(**cfg.adsorbate.matcher.model_dump())
    pure_atoms = build_pure_surfaces(cfg=cfg)
    index_selector_fn = naive_surface_index_selector
    swap_indices = index_selector_fn(cfg=cfg)
    surface_generator = mutate_via_swaps
    atoms_list = surface_generator(
        cfg=cfg,
        swap_indices=swap_indices,
        pure_atoms=pure_atoms,
        matcher=matcher,
    )
    adsorbate = Molecule(
        species=cfg.adsorbate.species,
        coords=cfg.adsorbate.coords,
        site_properties={
            "tags": [cfg.adsorbate.tag for _ in range(len(cfg.adsorbate.coords))]
        },
    )
    atoms_list = get_adsorbate_structures(
        cfg=cfg, atoms_list=atoms_list, adsorbate=adsorbate, matcher=matcher
    )
    config_db = connect(cfg.database.path)
    for atoms in atoms_list:
        config_db.write(atoms)
    print(f"Wrote {len(atoms_list)} structures to {cfg.database.path}")


if __name__ == "__main__":
    main()
