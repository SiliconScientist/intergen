# Installation and development

[Documentation index](README.md)

Use Python 3.11 or newer: although `pyproject.toml` declares Python 3.10+, the configuration loader imports the standard-library `tomllib`, available from Python 3.11.

From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The requirements file installs ASE, NumPy, pymatgen, TOML, Pydantic, and the local editable package. It also requests a `structure` extra that the project does not declare; pip may warn about it. The preceding dependency lines supply the runtime packages. `pip install -e .` alone is insufficient because the project’s dependency list is empty. The implementation also imports pandas, supplied transitively by pymatgen in the tested environment.

The editable install makes `python -m intergen` available in the separate run directory used by the quickstart. The lattice-constant asset is accessed relative to the working directory, rather than loaded as installed package data.

Run the existing tests from the repository root:

```sh
PYTHONPATH=src python -m unittest discover -s tests
```

The tests exercise metadata serialization, substitution provenance, adsorption matching, template behavior, and constraints. They do not establish completeness of arbitrary chemical searches or physical stability of generated structures.

## Implementation map

| File | Responsibility |
| --- | --- |
| [config.py](../src/intergen/config.py) | Pydantic models and fixed-name TOML loading. |
| [surface.py](../src/intergen/surface.py) | Slab construction, substitution plans, symmetry reduction, motif labels. |
| [adsorbate.py](../src/intergen/adsorbate.py) | Site discovery, placement, matching, template reuse, timing output. |
| [metadata.py](../src/intergen/metadata.py) | Provenance fields, normalization, database serialization. |
| [constraints.py](../src/intergen/constraints.py) | Geometric layer identification and ASE fixed-atom constraints. |
| [__main__.py](../src/intergen/__main__.py) | Generation workflow, database reuse, and output. |
