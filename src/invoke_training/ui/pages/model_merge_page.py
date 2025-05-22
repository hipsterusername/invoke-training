import logging

import gradio as gr

from invoke_training._shared.accelerator.accelerator_utils import get_dtype_from_str
from invoke_training._shared.stable_diffusion.model_loading_utils import PipelineVersionEnum
from invoke_training.model_merge.scripts.merge_models import parse_model_args, run_merge_models
from invoke_training.ui.config_groups.model_merge_config_group import MergeModelsConfigGroup
from invoke_training.ui.gradio_blocks.header import Header


class ModelMergePage:
    def __init__(self):
        self._merge_process_logger = logging.getLogger(__name__)
        theme = gr.themes.Default()
        theme._dark_mode = True
        custom_css = """
        .dark {
            --color-accent: #e6fd13 !important;
            --color-accent-soft: #e6fd1333 !important;
        }
        .dark .tabs button[aria-selected="true"] {
            color: #e6fd13 !important;
        }
        .dark input[type="checkbox"]:checked + span svg path {
            stroke: black !important;
        }
        """
        with gr.Blocks(
            theme=theme,
            css=custom_css,
            title="invoke-training",
            analytics_enabled=False,
            head='<link rel="icon" type="image/x-icon" href="/assets/favicon.png">',
        ) as app:
            self._header = Header()
            gr.Markdown("# Model Merge")
            self._config_group = MergeModelsConfigGroup()
            self._run_button = gr.Button("Run Merge")
            self._status = gr.Textbox(label="Status", interactive=False)

            self._run_button.click(
                self._on_run_merge_click,
                inputs=set(self._config_group.get_ui_input_components()),
                outputs=[self._status],
            )
        self._app = app

    def app(self):
        return self._app

    def _on_run_merge_click(self, data: dict):
        try:
            config = self._config_group.update_config_with_ui_component_data(None, data)
            merge_models = parse_model_args(config.models, [str(w) for w in config.weights])
            run_merge_models(
                logger=self._merge_process_logger,
                model_type=PipelineVersionEnum(config.model_type),
                models=merge_models,
                method=config.method,
                out_dir=config.out_dir,
                dtype=get_dtype_from_str(config.dtype),
            )
            return {self._status: "Merge completed."}
        except Exception as e:  # noqa: BLE001
            return {self._status: f"Error: {e}"}
