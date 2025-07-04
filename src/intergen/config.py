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


class Config(BaseModel):
    structure: StructureConfig
    generation: GenerationConfig
    database: DatabaseConfig
    adsorbate: AdsorbateConfig


def get_config():
    with open("config.toml", "rb") as f:
        cfg_data = load(f)
    return Config(**cfg_data)
