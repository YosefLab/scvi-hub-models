import logging

import click

from scvi_hub_models.config import json_data_store

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option("--model_name", type=str, help="Name of the model to run.")
@click.option("--dry_run", type=bool, default=False, help="Dry run the workflow.")
@click.option("--config_key", type=str, help="Use a different config file, e.g. for test purpose.")
@click.option("--save_dir", type=str, help="Directory to save intermediate results (defaults temporary).")
@click.option("--reload_data", type=bool, help="Reload the data or get from DVC.")
@click.option("--reload_model", type=bool, help="Reload the model or get from DVC.")
def run_workflow(
    model_name: str,
    dry_run: bool,
    config_key: str = None,
    save_dir: str = None,
    reload_data: bool = False,
    reload_model: bool = False) -> None:
    """Run the workflow for a specific model."""
    from importlib import import_module
    if not config_key:
        config_key = model_name

    workflow_module = import_module(f"scvi_hub_models.models._{model_name}")
    Workflow = workflow_module._Workflow
    config = json_data_store[config_key]

    workflow = Workflow(save_dir=save_dir, dry_run=dry_run, config=config, reload_data=reload_data, reload_model=reload_model)
    workflow.run()


if __name__ == "__main__":
    run_workflow()
