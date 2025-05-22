import gradio as gr

from invoke_training.config.model_merge.merge_models_config import MergeModelsConfig
from invoke_training.model_merge.utils.parse_model_arg import parse_model_arg

from .ui_config_element import UIConfigElement


class MergeModelsConfigGroup(UIConfigElement):
    def __init__(self):
        with gr.Row():
            self.model_type = gr.Dropdown(label="Model Type", choices=["SD", "SDXL"], value="SDXL", interactive=True)
            self.method = gr.Dropdown(label="Merge Method", choices=["LERP", "SLERP"], value="LERP", interactive=True)
        self.models_table = gr.Dataframe(
            headers=["Model", "Variant", "Weight"],
            datatype=["str", "str", "number"],
            row_count=2,
            col_count=3,
            interactive=True,
            label="Models",
        )
        with gr.Row():
            self.out_dir = gr.Textbox(label="Output Directory", interactive=True)
            self.dtype = gr.Dropdown(
                label="dtype",
                choices=["float32", "float16", "bfloat16"],
                value="float16",
                interactive=True,
            )

    def update_ui_components_with_config_data(
        self, config: MergeModelsConfig
    ) -> dict[gr.components.Component, list | str]:
        rows = []
        for model, weight in zip(config.models, config.weights, strict=True):
            model_name, variant = parse_model_arg(model)
            rows.append([model_name, variant or "", weight])
        return {
            self.model_type: config.model_type,
            self.method: config.method,
            self.models_table: rows,
            self.out_dir: config.out_dir,
            self.dtype: config.dtype,
        }

    def update_config_with_ui_component_data(self, orig_config: MergeModelsConfig, ui_data: dict) -> MergeModelsConfig:
        rows = ui_data.pop(self.models_table)
        models: list[str] = []
        weights: list[float] = []
        for row in rows:
            if not row or row[0] is None or row[0] == "":
                continue
            model = row[0].strip()
            variant = str(row[1]).strip() if row[1] else None
            weight = float(row[2]) if row[2] is not None else 1.0
            models.append(f"{model}::{variant}" if variant else model)
            weights.append(weight)
        new_config = MergeModelsConfig(
            model_type=ui_data.pop(self.model_type),
            method=ui_data.pop(self.method),
            models=models,
            weights=weights,
            out_dir=ui_data.pop(self.out_dir),
            dtype=ui_data.pop(self.dtype),
        )
        return new_config
