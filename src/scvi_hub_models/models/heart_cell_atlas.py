from anndata import AnnData
from scvi.hub import HubModel
from scvi.model import SCVI


def load_adata(save_dir: str) -> AnnData:
    """Download and load the dataset."""
    from scvi.data import heart_cell_atlas_subsampled

    return heart_cell_atlas_subsampled(save_path=save_dir)


def preprocess_adata(adata: AnnData) -> AnnData:
    """Preprocess the AnnData object."""
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


def initialize_model(adata: AnnData) -> SCVI:
    """Initialize the scVI model."""
    import scvi

    scvi.settings.seed = 0
    SCVI.setup_anndata(
        adata,
        layer="counts",
        categorical_covariate_keys=["cell_source", "donor"],
        continuous_covariate_keys=["percent_mito", "percent_ribo"],
    )
    return SCVI(adata)


def train_model(model: SCVI) -> SCVI:
    """Train the scVI model."""
    import scvi
    import torch

    scvi.settings.seed = 0
    torch.set_float32_matmul_precision("high")
    model.train()

    return model


def save_model(model: SCVI, config: dict, save_dir: str) -> str:
    """Save the scVI model."""
    import os

    model_path = os.path.join(save_dir, config["model_dir"])
    model.save(model_path, save_anndata=True, overwrite=True)

    return model_path


def create_hub_model(model_path: str, config: dict) -> HubModel:
    """Create a HubModel from the scVI model."""
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


def model_workflow(dry_run: bool, repo_create: bool):
    """Run the model workflow."""
    from tempfile import TemporaryDirectory

    from scvi_hub_models.config import HEART_CELL_ATLAS_CONFIG
    from scvi_hub_models.utils import upload_hub_model

    config = HEART_CELL_ATLAS_CONFIG
    save_dir = TemporaryDirectory().name

    # load and preprocess the data
    adata = load_adata(save_dir)
    adata = preprocess_adata(adata)

    # train the model
    model = initialize_model(adata)
    model = train_model(model, adata)
    model_path = save_model(model, config, save_dir)

    # create and upload to hub
    hub_model = create_hub_model(model_path, config)
    hub_model = upload_hub_model(hub_model, config["repo_name"])
