from __future__ import annotations

import logging
import os

from anndata import AnnData
from scvi.hub import HubModel
from scvi.model import SCVI

logger = logging.getLogger(__name__)


def _load_adata(save_dir: str) -> AnnData:
    from scvi.data import heart_cell_atlas_subsampled

    return heart_cell_atlas_subsampled(save_path=save_dir)


def load_adata(save_dir: str, dry_run: bool) -> AnnData | None:
    """Download and load the dataset."""
    logger.info(f"Saving heart cell atlas dataset to {save_dir}.")
    return _load_adata(save_dir) if not dry_run else None


def _preprocess_adata(adata: AnnData) -> AnnData:
    import scanpy as sc

    sc.pp.filter_genes(adata, min_counts=3)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=1200,
        subset=True,
        layer="counts",
        flavor="seurat_v3",
        batch_key="cell_source",
    )

    return adata


def preprocess_adata(adata: AnnData, dry_run: bool) -> AnnData:
    """Preprocess the AnnData object."""
    logger.info("Preprocessing the AnnData object.")
    return _preprocess_adata(adata) if not dry_run else adata


def _initialize_model(adata: AnnData) -> SCVI:
    SCVI.setup_anndata(
        adata,
        layer="counts",
        categorical_covariate_keys=["cell_source", "donor"],
        continuous_covariate_keys=["percent_mito", "percent_ribo"],
    )
    return SCVI(adata)


def initialize_model(adata: AnnData, dry_run: bool) -> SCVI | None:
    """Initialize the scVI model."""
    logger.info("Initializing the scVI model.")
    return _initialize_model(adata) if not dry_run else None


def _train_model(model: SCVI, save_dir: str, seed: int, config: dict) -> SCVI:
    """Train the scVI model."""
    from lightning.pytorch.loggers import TensorBoardLogger
    from scvi import settings

    settings.seed = seed
    os.path.join(save_dir, config["model_dir"])
    TensorBoardLogger()
    model.train()

    return model


def train_model(model: SCVI, save_dir: str, seed: int, config: dict, dry_run: bool) -> SCVI | None:
    """Train the scVI model."""
    logger.info("Training the scVI model.")
    return _train_model(model, save_dir, seed, config) if not dry_run else model


def _save_model(model: SCVI, config: dict, save_dir: str) -> str:
    model_path = os.path.join(save_dir, config["model_dir"])
    model.save(model_path, save_anndata=True, overwrite=True)

    return model_path


def save_model(model: SCVI, config: dict, save_dir: str, dry_run: bool) -> str | None:
    """Save the scVI model."""
    logger.info(f"Saving the scVI model to {os.path.join(save_dir, config['model_dir'])}.")
    return _save_model(model, config, save_dir) if not dry_run else None


def _create_hub_model(model_path: str, config: dict) -> HubModel:
    import anndata
    from scvi.hub import HubMetadata, HubModel, HubModelCardHelper

    metadata = config["metadata"]

    hub_metadata = HubMetadata.from_dir(
        model_path,
        training_data_url=metadata["training_data_url"],
        anndata_version=anndata.__version__,
    )
    model_card = HubModelCardHelper.from_dir(
        model_path,
        training_data_url=metadata["training_data_url"],
        tissues=metadata["tissues"],
        data_modalities=metadata["data_modalities"],
        description=metadata["description"],
        references=metadata["references"],
        license_info=metadata["license_info"],
        data_is_annotated=metadata["data_is_annotated"],
        anndata_version=anndata.__version__,
        data_is_minified=False,
    )
    return HubModel(
        model_path,
        metadata=hub_metadata,
        model_card=model_card,
    )


def create_hub_model(model_path: str, config: dict, dry_run: bool) -> HubModel | None:
    """Create a HubModel from the scVI model."""
    logger.info(f"Creating a HubModel from {model_path}.")
    return _create_hub_model(model_path, config) if not dry_run else None


def model_workflow(dry_run: bool, repo_create: bool):
    """Run the model workflow."""
    from tempfile import TemporaryDirectory

    from scvi_hub_models.config import HEART_CELL_ATLAS_CONFIG
    from scvi_hub_models.utils import upload_hub_model

    config = HEART_CELL_ATLAS_CONFIG
    save_dir = TemporaryDirectory().name

    adata = load_adata(save_dir, dry_run)
    adata = preprocess_adata(adata, dry_run)

    model = initialize_model(adata, dry_run)
    model = train_model(model, save_dir, 0, config, dry_run)
    model_path = save_model(model, config, save_dir, dry_run)

    hub_model = create_hub_model(model_path, config, dry_run)
    hub_model = upload_hub_model(hub_model, config["repo_name"], dry_run, repo_create=repo_create)
