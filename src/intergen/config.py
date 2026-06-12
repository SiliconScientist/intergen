from pathlib import Path
from typing import Any, Literal, Union, Optional

from pydantic import BaseModel
from tomllib import load


class StructureConfig(BaseModel):
    hcp_list: list[str]
    fcc_list: list[str]
    size: tuple[int, int, int]
    vacuum: float


class GenerationConfig(BaseModel):
    layers_to_swap: int
    num_swaps: int
    swap_elements: list[str]
    only_last_generation: bool


class DatabaseConfig(BaseModel):
    path: Path
    constrain_bottom_layers: int = 0
    constraint_z_tolerance: float = 0.5
    constraint_lowest_z_tolerance: float = 0.5


class MatcherConfig(BaseModel):
    ltol: float
    stol: float
    angle_tol: float


class AdsorbateConfig(BaseModel):
    matcher: MatcherConfig
    species: str
    coords: list[tuple[float, float, float]]
    sites: list[str]
    tag: int
    surface_layers_for_matching: int
    reuse_site_templates_for_two_swap_motifs: bool = True
    template_site_match_tolerance: float = 0.5


class Config(BaseModel):
    structure: StructureConfig
    generation: GenerationConfig
    database: DatabaseConfig
    adsorbate: AdsorbateConfig


def get_config():
    with open("config.toml", "rb") as f:
        cfg_data = load(f)
    return Config(**cfg_data)
