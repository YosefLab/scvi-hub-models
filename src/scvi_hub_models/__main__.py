import logging

import click

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option("--model_name", type=str, help="Name of the model to run.")
@click.option("--dry_run", type=bool, default=False, help="Dry run the workflow.")
def run_workflow(model_name: str, dry_run: bool) -> None:
    """Run the workflow for a specific model."""
    from importlib import import_module

    _model_name = model_name.replace("-", "_")  # e.g. "test-scvi" -> "test_scvi"
    workflow_module = import_module(f"scvi_hub_models.models._{_model_name}")
    config_module = import_module(f"scvi_hub_models.config._{_model_name}")
    Workflow = workflow_module._Workflow
    CONFIG = config_module._CONFIG

    workflow = Workflow(dry_run=dry_run, config=CONFIG)
    workflow.run()


if __name__ == "__main__":
    run_workflow()
