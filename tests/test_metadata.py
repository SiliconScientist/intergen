import unittest

from intergen.metadata import (
    ADSLAB_ID_KEY,
    ADSORBATE_KEY,
    ADSORPTION_SITE_BRIDGE,
    ADSORPTION_SITE_FCC_HOLLOW,
    ADSORPTION_SITE_HCP_HOLLOW,
    ADSORPTION_SITE_LABELS,
    ADSORPTION_SITE_TOP,
    HOST_ELEMENT_KEY,
    INITIAL_SITE_COORDINATE_KEY,
    INITIAL_SITE_LABEL_KEY,
    PARENT_SLAB_ID_KEY,
    SLAB_ID_KEY,
    STRUCTURE_METADATA_FIELDS,
    SUPERCELL_SIZE_KEY,
    SURFACE_TYPE_FCC111,
    SURFACE_TYPE_HCP0001,
    SURFACE_TYPE_KEY,
    SURFACE_TYPES,
    SWAP_ELEMENTS_KEY,
    SWAP_INDICES_KEY,
    TOP_LAYER_MOTIF_DUAL_SINGLE_ATOM_ALLOY,
    TOP_LAYER_MOTIF_HETERODIMER,
    TOP_LAYER_MOTIF_KEY,
    TOP_LAYER_MOTIF_PURE,
    TOP_LAYER_MOTIF_SINGLE_SWAP,
    TOP_LAYER_MOTIFS,
    normalize_adsorption_site_label,
    normalize_site_coordinate,
    validate_structure_metadata_keys,
)


class TestMetadataSchema(unittest.TestCase):
    def test_structure_metadata_fields_match_expected_schema(self):
        self.assertEqual(
            STRUCTURE_METADATA_FIELDS,
            (
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
            ),
        )

    def test_surface_types_are_normalized(self):
        self.assertEqual(SURFACE_TYPES, (SURFACE_TYPE_FCC111, SURFACE_TYPE_HCP0001))

    def test_top_layer_motifs_are_normalized(self):
        self.assertEqual(
            TOP_LAYER_MOTIFS,
            (
                TOP_LAYER_MOTIF_PURE,
                TOP_LAYER_MOTIF_SINGLE_SWAP,
                TOP_LAYER_MOTIF_HETERODIMER,
                TOP_LAYER_MOTIF_DUAL_SINGLE_ATOM_ALLOY,
            ),
        )

    def test_adsorption_site_labels_are_normalized(self):
        self.assertEqual(
            ADSORPTION_SITE_LABELS,
            (
                ADSORPTION_SITE_TOP,
                ADSORPTION_SITE_BRIDGE,
                ADSORPTION_SITE_FCC_HOLLOW,
                ADSORPTION_SITE_HCP_HOLLOW,
            ),
        )

    def test_validate_structure_metadata_keys_accepts_known_keys(self):
        validate_structure_metadata_keys(
            {
                SLAB_ID_KEY: "slab-0001",
                INITIAL_SITE_LABEL_KEY: ADSORPTION_SITE_FCC_HOLLOW,
                ADSORBATE_KEY: "OH",
            }
        )

    def test_validate_structure_metadata_keys_rejects_unknown_keys(self):
        with self.assertRaisesRegex(
            ValueError, "Unknown structure metadata keys"
        ):
            validate_structure_metadata_keys({"site_label": "top"})


class TestMetadataNormalization(unittest.TestCase):
    def test_normalize_adsorption_site_label_accepts_aliases(self):
        self.assertEqual(normalize_adsorption_site_label("top"), ADSORPTION_SITE_TOP)
        self.assertEqual(
            normalize_adsorption_site_label("fcc hollow"),
            ADSORPTION_SITE_FCC_HOLLOW,
        )
        self.assertEqual(
            normalize_adsorption_site_label("hcp_hollow"),
            ADSORPTION_SITE_HCP_HOLLOW,
        )
        self.assertEqual(
            normalize_adsorption_site_label("Bridge"),
            ADSORPTION_SITE_BRIDGE,
        )

    def test_normalize_adsorption_site_label_rejects_unknown_label(self):
        with self.assertRaisesRegex(
            ValueError, "Unsupported adsorption site label"
        ):
            normalize_adsorption_site_label("atop")

    def test_normalize_site_coordinate_returns_float_triplet(self):
        self.assertEqual(
            normalize_site_coordinate((1, 2.5, 3)),
            (1.0, 2.5, 3.0),
        )

    def test_normalize_site_coordinate_requires_three_values(self):
        with self.assertRaisesRegex(
            ValueError, "must contain exactly three values"
        ):
            normalize_site_coordinate((0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
