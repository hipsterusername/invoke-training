from typing import Literal

from invoke_training.config.config_base_model import ConfigBaseModel


class MergeModelsConfig(ConfigBaseModel):
    """Configuration for merging multiple base models."""

    type: Literal["MERGE_MODELS"] = "MERGE_MODELS"
    model_type: Literal["SD", "SDXL"]
    models: list[str]
    weights: list[float]
    method: Literal["LERP", "SLERP"] = "LERP"
    out_dir: str
    dtype: Literal["float32", "float16", "bfloat16"] = "float16"
