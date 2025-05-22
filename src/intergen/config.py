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
    num_swaps: int
    layers_to_swap: int
    swap_elements: list[str]


class DatabaseConfig(BaseModel):
    path: Path


class AdsorbateConfig(BaseModel):
    species: str
    coords: list[tuple[float, float, float]]
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
