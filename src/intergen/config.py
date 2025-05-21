from pathlib import Path
from typing import Any, Literal, Union, Optional

from pydantic import BaseModel
from tomli import load


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


class Config(BaseModel):
    structure: StructureConfig
    generation: GenerationConfig
    database: DatabaseConfig


def get_config():
    with open("config.toml", "rb") as f:
        cfg_data = load(f)
    return Config(**cfg_data)
