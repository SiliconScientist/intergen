from pathlib import Path
from itertools import count

from ase.db import connect
from ase.visualize import view
from pymatgen.core import Molecule
from pymatgen.analysis.structure_matcher import StructureMatcher
from intergen.config import get_config
from intergen.surface import (
    build_pure_surfaces,
    iterative_swaps,
    get_metadata_atoms_per_layer,
    get_swap_plans,
)
from intergen.adsorbate import get_adsorbate_structures
from intergen.constraints import (
    constrain_adsorbate_bottom_layers,
    has_adsorbate_bottom_layer_constraints,
)
from intergen.metadata import (
    DB_METADATA_DATA_KEY,
    deserialize_structure_metadata_from_db,
    get_structure_metadata,
    serialize_structure_metadata_for_db,
)


def warn_if_constraints_disabled(cfg) -> None:
    if cfg.database.constrain_bottom_layers > 0:
        return
    print(
        "WARNING: database.constrain_bottom_layers is 0, so no FixAtoms constraints "
        f"will be written to {cfg.database.path}."
    )


def get_initial_atoms_list(pure_atoms, only_last_generation):
    if only_last_generation:
        return []
    return list(pure_atoms)


def apply_database_constraints(cfg, atoms_list):
    bottom_layers = cfg.database.constrain_bottom_layers
    if bottom_layers <= 0:
        return list(atoms_list)

    adsorbate_len = len(cfg.adsorbate.coords)
    constrained_atoms_list = []
    for atoms in atoms_list:
        if has_adsorbate_bottom_layer_constraints(
            atoms,
            adsorbate_len=adsorbate_len,
            bottom_layers=bottom_layers,
            z_tolerance=cfg.database.constraint_z_tolerance,
            lowest_z_tolerance=cfg.database.constraint_lowest_z_tolerance,
        ):
            constrained_atoms_list.append(atoms)
            continue
        constrained_atoms_list.append(
            constrain_adsorbate_bottom_layers(
                atoms,
                adsorbate_len=adsorbate_len,
                bottom_layers=bottom_layers,
                z_tolerance=cfg.database.constraint_z_tolerance,
                lowest_z_tolerance=cfg.database.constraint_lowest_z_tolerance,
            )
        )
    return constrained_atoms_list


def write_atoms_database(path: Path, atoms_list) -> None:
    if path.exists():
        path.unlink()
    config_db = connect(path)
    for atoms in atoms_list:
        structure_metadata = get_structure_metadata(atoms.info)
        config_db.write(
            atoms,
            key_value_pairs=serialize_structure_metadata_for_db(structure_metadata),
            data={DB_METADATA_DATA_KEY: structure_metadata},
        )


def get_database_atoms_list(path: Path):
    config_db = connect(path)
    atoms_list = []
    for row in config_db.select():
        atoms = row.toatoms()
        atoms.info.update(
            deserialize_structure_metadata_from_db(
                key_value_pairs=row.key_value_pairs,
                data=row.data,
            )
        )
        atoms_list.append(atoms)
    return atoms_list


def main():
    cfg = get_config()
    warn_if_constraints_disabled(cfg)
    if cfg.database.path.exists():
        atoms_list = get_database_atoms_list(cfg.database.path)
        if atoms_list:
            constrained_atoms_list = apply_database_constraints(cfg=cfg, atoms_list=atoms_list)
            if all(
                before is after for before, after in zip(atoms_list, constrained_atoms_list)
            ):
                print(f"Output already exists at {cfg.database.path}. Skipping generation.")
                return
            write_atoms_database(cfg.database.path, constrained_atoms_list)
            print(
                f"Wrote {len(constrained_atoms_list)} constrained structures to "
                f"{cfg.database.path}"
            )
            return
    matcher = StructureMatcher(**cfg.adsorbate.matcher.model_dump())
    surface_generator = iterative_swaps
    slab_id_source = count(1)
    pure_atoms = build_pure_surfaces(cfg=cfg, slab_id_source=slab_id_source)
    atoms_list = get_initial_atoms_list(
        pure_atoms=pure_atoms,
        only_last_generation=cfg.generation.only_last_generation,
    )
    for atoms in pure_atoms:
        atoms_per_layer = get_metadata_atoms_per_layer(atoms)
        swap_indices = list(range(atoms_per_layer * cfg.generation.layers_to_swap))
        host_element = atoms.get_chemical_symbols()[0]
        swap_plans = get_swap_plans(
            cfg=cfg,
            num_swaps=cfg.generation.num_swaps,
            host_element=host_element,
        )
        for swap_plan in swap_plans:
            new_atoms_list = surface_generator(
                atoms=atoms,
                host_element=host_element,
                swap_plan=swap_plan,
                indices=swap_indices,
                atoms_per_layer=atoms_per_layer,
                slab_id_source=slab_id_source,
                matcher=matcher,
                only_last_generation=cfg.generation.only_last_generation,
            )
            atoms_list.extend(new_atoms_list)
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
    atoms_list = apply_database_constraints(cfg=cfg, atoms_list=atoms_list)
    write_atoms_database(cfg.database.path, atoms_list)
    print(f"Wrote {len(atoms_list)} structures to {cfg.database.path}")


if __name__ == "__main__":
    main()
