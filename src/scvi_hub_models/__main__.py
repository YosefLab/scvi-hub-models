from scvi_hub_models.utils import wrap_kwargs


@wrap_kwargs
def run_workflow(
    model_name: str,
    dry_run: bool = False,
    repo_create: bool = False,
) -> None:
    """Run the workflow for a specific model."""
    import logging

    logger = logging.getLogger(__name__)

    if model_name == "heart_cell_atlas":
        from scvi_hub_models.models.heart_cell_atlas import model_workflow
    elif model_name == "human_lung_cell_atlas":
        from scvi_hub_models.models.human_lung_cell_atlas import model_workflow
    elif model_name == "tabula_sapiens":
        from scvi_hub_models.models.tabula_sapiens import model_workflow

    logger.info(f"Started running {model_name} workflow with `dry_run={dry_run}` and " f"`repo_create={repo_create}`.")
    model_workflow(dry_run=dry_run, repo_create=repo_create)


if __name__ == "__main__":
    run_workflow()
