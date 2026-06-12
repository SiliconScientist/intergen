from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms


def best_fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    points = np.asarray(points, dtype=float)
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    d = -float(np.dot(normal, centroid))
    distances = centered @ normal
    rms = float(np.sqrt(np.mean(distances**2)))
    return centroid, normal, d, rms


def rewrap_slab_by_largest_gap(atoms: Atoms) -> Atoms:
    if len(atoms) == 0:
        return atoms.copy()

    rewrapped = atoms.copy()
    scaled_positions = rewrapped.get_scaled_positions(wrap=True)
    fractional_z = np.mod(scaled_positions[:, 2], 1.0)
    sorted_fractional_z = np.sort(fractional_z)
    cyclic_gaps = np.diff(
        np.concatenate((sorted_fractional_z, [sorted_fractional_z[0] + 1.0]))
    )
    largest_gap_index = int(np.argmax(cyclic_gaps))
    gap_start = sorted_fractional_z[largest_gap_index]
    gap_midpoint = (gap_start + 0.5 * cyclic_gaps[largest_gap_index]) % 1.0

    scaled_positions[:, 2] = np.mod(fractional_z - gap_midpoint, 1.0)
    rewrapped.set_scaled_positions(scaled_positions)
    return rewrapped


def get_lowest_atom_indices(atoms: Atoms, z_tolerance: float = 0.5) -> list[int]:
    z_values = atoms.get_positions()[:, 2]
    z_min = float(np.min(z_values))
    return np.where(np.abs(z_values - z_min) <= z_tolerance)[0].tolist()


def plane_from_lowest_atoms(
    atoms: Atoms, lowest_z_tolerance: float = 0.5
) -> tuple[np.ndarray, np.ndarray, float]:
    working = rewrap_slab_by_largest_gap(atoms)
    repeated_atoms = working.repeat(rep=(2, 2, 1))
    lowest_indices = get_lowest_atom_indices(
        repeated_atoms, z_tolerance=lowest_z_tolerance
    )
    if len(lowest_indices) < 3:
        raise ValueError(
            'At least 3 atoms are required in the lowest group to define a plane'
        )

    lowest_points = repeated_atoms.get_positions()[lowest_indices]
    centroid, normal, d, _ = best_fit_plane(lowest_points)

    bottom_centroid = lowest_points.mean(axis=0)
    all_centroid = repeated_atoms.get_positions().mean(axis=0)
    if float(np.dot(normal, all_centroid - bottom_centroid)) < 0.0:
        normal = -normal
        d = -d
    return centroid, normal, d


def get_layer_indices(
    atoms: Atoms,
    z_tolerance: float = 0.5,
    lowest_z_tolerance: float = 0.5,
) -> dict[int, list[int]]:
    working = rewrap_slab_by_largest_gap(atoms)
    _, normal, d = plane_from_lowest_atoms(
        working, lowest_z_tolerance=lowest_z_tolerance
    )
    heights = working.get_positions() @ normal + d
    sorted_indices = np.argsort(heights)
    sorted_heights = heights[sorted_indices]

    layers: list[list[int]] = []
    current_layer = [int(sorted_indices[0])]
    for index in range(1, len(working)):
        if abs(float(sorted_heights[index] - sorted_heights[index - 1])) <= z_tolerance:
            current_layer.append(int(sorted_indices[index]))
        else:
            layers.append(current_layer)
            current_layer = [int(sorted_indices[index])]
    layers.append(current_layer)

    return {layer_index + 1: layer for layer_index, layer in enumerate(layers)}


def get_indices_for_layers(
    atoms: Atoms,
    layers: int | Iterable[int],
    z_tolerance: float = 0.5,
    lowest_z_tolerance: float = 0.5,
) -> list[int]:
    if isinstance(layers, int):
        requested_layers = [layers]
    else:
        requested_layers = list(layers)

    layer_indices = get_layer_indices(
        atoms,
        z_tolerance=z_tolerance,
        lowest_z_tolerance=lowest_z_tolerance,
    )
    layer_count = len(layer_indices)
    indices: list[int] = []
    for layer in requested_layers:
        resolved_layer = layer_count + layer + 1 if layer < 0 else layer
        if resolved_layer not in layer_indices:
            raise ValueError(
                f'Requested layer {layer}, but only layers 1..{layer_count} or '
                f'-1..{-layer_count} exist'
            )
        indices.extend(layer_indices[resolved_layer])
    return sorted(indices)


def constrain_adsorbate_bottom_layers(
    atoms: Atoms,
    adsorbate_len: int,
    bottom_layers: int = 2,
    z_tolerance: float = 0.5,
    lowest_z_tolerance: float = 0.5,
) -> Atoms:
    if bottom_layers <= 0:
        return atoms.copy()
    if adsorbate_len < 0 or adsorbate_len > len(atoms):
        raise ValueError('adsorbate_len must be between 0 and len(atoms)')

    slab_atom_count = len(atoms) - adsorbate_len
    if slab_atom_count <= 0:
        raise ValueError('Cannot constrain slab layers without slab atoms')

    slab = atoms[:slab_atom_count].copy()
    fixed_indices = get_indices_for_layers(
        slab,
        layers=tuple(range(1, bottom_layers + 1)),
        z_tolerance=z_tolerance,
        lowest_z_tolerance=lowest_z_tolerance,
    )

    constrained = atoms.copy()
    existing = constrained.constraints
    if existing is None:
        constraints = []
    elif isinstance(existing, (list, tuple)):
        constraints = list(existing)
    else:
        constraints = [existing]
    constraints.append(FixAtoms(indices=fixed_indices))
    constrained.set_constraint(constraints)
    return constrained
