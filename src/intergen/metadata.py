import json
from collections.abc import Mapping, Sequence
from typing import Final


SLAB_ID_KEY: Final[str] = "slab_id"
ADSLAB_ID_KEY: Final[str] = "adslab_id"
PARENT_SLAB_ID_KEY: Final[str] = "parent_slab_id"
HOST_ELEMENT_KEY: Final[str] = "host_element"
SURFACE_TYPE_KEY: Final[str] = "surface_type"
SUPERCELL_SIZE_KEY: Final[str] = "supercell_size"
SWAP_INDICES_KEY: Final[str] = "swap_indices"
SWAP_ELEMENTS_KEY: Final[str] = "swap_elements"
TOP_LAYER_MOTIF_KEY: Final[str] = "top_layer_motif"
INITIAL_SITE_LABEL_KEY: Final[str] = "initial_site_label"
INITIAL_SITE_COORDINATE_KEY: Final[str] = "initial_site_coordinate"
ADSORBATE_KEY: Final[str] = "adsorbate"

STRUCTURE_METADATA_FIELDS: Final[tuple[str, ...]] = (
    SLAB_ID_KEY,
    ADSLAB_ID_KEY,
    PARENT_SLAB_ID_KEY,
    HOST_ELEMENT_KEY,
    SURFACE_TYPE_KEY,
    SUPERCELL_SIZE_KEY,
    SWAP_INDICES_KEY,
    SWAP_ELEMENTS_KEY,
    TOP_LAYER_MOTIF_KEY,
    INITIAL_SITE_LABEL_KEY,
    INITIAL_SITE_COORDINATE_KEY,
    ADSORBATE_KEY,
)

SLAB_PROVENANCE_FIELDS: Final[tuple[str, ...]] = (
    SLAB_ID_KEY,
    HOST_ELEMENT_KEY,
    SURFACE_TYPE_KEY,
    SUPERCELL_SIZE_KEY,
    SWAP_INDICES_KEY,
    SWAP_ELEMENTS_KEY,
    TOP_LAYER_MOTIF_KEY,
)

DB_METADATA_DATA_KEY: Final[str] = "structure_metadata"

SURFACE_TYPE_FCC111: Final[str] = "fcc111"
SURFACE_TYPE_HCP0001: Final[str] = "hcp0001"
SURFACE_TYPES: Final[tuple[str, ...]] = (
    SURFACE_TYPE_FCC111,
    SURFACE_TYPE_HCP0001,
)

TOP_LAYER_MOTIF_PURE: Final[str] = "pure"
TOP_LAYER_MOTIF_SINGLE_SWAP: Final[str] = "single_swap"
TOP_LAYER_MOTIF_HETERODIMER: Final[str] = "heterodimer"
TOP_LAYER_MOTIF_DUAL_SINGLE_ATOM_ALLOY: Final[str] = "dual_single_atom_alloy"
TOP_LAYER_MOTIFS: Final[tuple[str, ...]] = (
    TOP_LAYER_MOTIF_PURE,
    TOP_LAYER_MOTIF_SINGLE_SWAP,
    TOP_LAYER_MOTIF_HETERODIMER,
    TOP_LAYER_MOTIF_DUAL_SINGLE_ATOM_ALLOY,
)

ADSORPTION_SITE_TOP: Final[str] = "top"
ADSORPTION_SITE_BRIDGE: Final[str] = "bridge"
ADSORPTION_SITE_HOLLOW: Final[str] = "hollow"
ADSORPTION_SITE_FCC_HOLLOW: Final[str] = "fcc_hollow"
ADSORPTION_SITE_HCP_HOLLOW: Final[str] = "hcp_hollow"
ADSORPTION_SITE_LABELS: Final[tuple[str, ...]] = (
    ADSORPTION_SITE_TOP,
    ADSORPTION_SITE_BRIDGE,
    ADSORPTION_SITE_HOLLOW,
    ADSORPTION_SITE_FCC_HOLLOW,
    ADSORPTION_SITE_HCP_HOLLOW,
)

_ADSORPTION_SITE_LABEL_ALIASES: Final[Mapping[str, str]] = {
    "top": ADSORPTION_SITE_TOP,
    "ontop": ADSORPTION_SITE_TOP,
    "bridge": ADSORPTION_SITE_BRIDGE,
    "hollow": ADSORPTION_SITE_HOLLOW,
    "fcc hollow": ADSORPTION_SITE_FCC_HOLLOW,
    "fcc_hollow": ADSORPTION_SITE_FCC_HOLLOW,
    "hcp hollow": ADSORPTION_SITE_HCP_HOLLOW,
    "hcp_hollow": ADSORPTION_SITE_HCP_HOLLOW,
}


def normalize_adsorption_site_label(label: str) -> str:
    normalized_label = label.strip().lower().replace("-", " ").replace("_", " ")
    if normalized_label not in _ADSORPTION_SITE_LABEL_ALIASES:
        raise ValueError(f"Unsupported adsorption site label: {label!r}")
    return _ADSORPTION_SITE_LABEL_ALIASES[normalized_label]


def normalize_site_coordinate(
    coordinate: Sequence[float | int],
) -> tuple[float, float, float]:
    if len(coordinate) != 3:
        raise ValueError(
            "Initial adsorption site coordinates must contain exactly three values."
        )
    return tuple(float(value) for value in coordinate)


def validate_structure_metadata_keys(metadata: Mapping[str, object]) -> None:
    unknown_keys = sorted(set(metadata) - set(STRUCTURE_METADATA_FIELDS))
    if unknown_keys:
        raise ValueError(f"Unknown structure metadata keys: {unknown_keys}")


def get_structure_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    return {key: metadata[key] for key in STRUCTURE_METADATA_FIELDS if key in metadata}


def serialize_structure_metadata_for_db(
    metadata: Mapping[str, object],
) -> dict[str, str | int | float | bool]:
    serialized_metadata: dict[str, str | int | float | bool] = {}
    for key, value in get_structure_metadata(metadata).items():
        if isinstance(value, (str, int, float, bool)):
            serialized_metadata[key] = value
            continue
        serialized_metadata[key] = json.dumps(value)
    return serialized_metadata


def deserialize_structure_metadata_from_db(
    key_value_pairs: Mapping[str, object],
    data: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if data is not None and DB_METADATA_DATA_KEY in data:
        metadata = get_structure_metadata(data[DB_METADATA_DATA_KEY])
        if SUPERCELL_SIZE_KEY in metadata:
            metadata[SUPERCELL_SIZE_KEY] = tuple(metadata[SUPERCELL_SIZE_KEY])
        if INITIAL_SITE_COORDINATE_KEY in metadata:
            metadata[INITIAL_SITE_COORDINATE_KEY] = normalize_site_coordinate(
                metadata[INITIAL_SITE_COORDINATE_KEY]
            )
        return dict(metadata)

    restored_metadata = {}
    for key in STRUCTURE_METADATA_FIELDS:
        if key not in key_value_pairs:
            continue
        value = key_value_pairs[key]
        if key in {
            SUPERCELL_SIZE_KEY,
            SWAP_INDICES_KEY,
            SWAP_ELEMENTS_KEY,
            INITIAL_SITE_COORDINATE_KEY,
        } and isinstance(value, str):
            restored_metadata[key] = json.loads(value)
            continue
        restored_metadata[key] = value
    if SUPERCELL_SIZE_KEY in restored_metadata:
        restored_metadata[SUPERCELL_SIZE_KEY] = tuple(
            restored_metadata[SUPERCELL_SIZE_KEY]
        )
    if INITIAL_SITE_COORDINATE_KEY in restored_metadata:
        restored_metadata[INITIAL_SITE_COORDINATE_KEY] = normalize_site_coordinate(
            restored_metadata[INITIAL_SITE_COORDINATE_KEY]
        )
    validate_structure_metadata_keys(restored_metadata)
    return restored_metadata
