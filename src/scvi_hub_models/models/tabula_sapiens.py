from anndata import AnnData
from scvi.model.base import BaseModelClass


def download_models_for_tissue(tissue: str, config: dict, save_dir: str) -> str:
    """Download the models for a given tissue from Zenodo.

    Returns the path to the directory containing the models.
    """
    from pathlib import Path

    from pooch import Untar, retrieve

    untarred = retrieve(
        url=f"{config['base_url']}{tissue}{config['models_suffix']}",
        known_hash=config["model_hashes"][tissue],
        fname=f"{tissue}_models",
        path=save_dir,
        processor=Untar(),
    )
    untarred = sorted(untarred)
    return str(Path(untarred[0]).parent.parent)


def download_models(config: dict, save_dir: str):
    """Download the models for a list of tissues from Zenodo."""
    base_model_dirs = {}
    for tissue in config["tissues"]:
        base_model_dirs[tissue] = download_models_for_tissue(tissue, config, save_dir)

    return base_model_dirs


def download_adata_for_tissue(tissue: str, config: dict, save_dir: str):
    """Download the dataset for a given tissue from Zenodo.

    Returns the path to the dataset.
    """
    from pooch import retrieve

    return retrieve(
        url=f"{config['base_url']}{tissue}{config['adata_suffix']}",
        known_hash=config["adata_hashes"][tissue],
        fname=f"{tissue}_adata.h5ad",
        path=save_dir,
        processor=None,
    )


def download_adatas(config: dict, save_dir: str):
    """Download the datasets for a list of tissues from Zenodo."""
    adata_paths = {}
    for tissue in config["tissues"]:
        adata_paths[tissue] = download_adata_for_tissue(tissue, config, save_dir)

    return adata_paths


def minify_and_save_model(model: BaseModelClass, adata: AnnData, save_dir: str) -> str:
    """Minify and save the model."""
    import os

    qzm, qzv = model.get_latent_representation(give_mean=False, return_dist=True)
    model_name = model.__class__.__name__
    qzm_key = f"{model_name}_latent_qzm"
    qzv_key = f"{model_name}_latent_qzv"
    adata.obsm[qzm_key] = qzm
    adata.obsm[qzv_key] = qzv
    model.minify_adata(use_latent_qzm_key=qzm_key, use_latent_qzv_key=qzv_key)
    mini_model_path = os.path.join(save_dir, f"mini_{model_name}")
    model.save(mini_model_path, overwrite=True, save_anndata=True)

    return mini_model_path


def create_hub_model(model_path: str, tissue: str, config: dict, minified: bool = False):
    """Create a HubModel from the model."""
    import anndata
    from scvi.hub import HubMetadata, HubModel, HubModelCardHelper

    metadata = config["metadata"]

    hub_metadata = HubMetadata.from_dir(
        model_path,
        anndata_version=anndata.__version__,
    )
    model_card = HubModelCardHelper.from_dir(
        model_path,
        training_data_url=f"{config['base_url']}{tissue}{config['adata_suffix']}",
        tissues=[tissue],
        data_modalities=metadata["data_modalities"],
        description=metadata["description"],
        references=metadata["references"],
        license_info=metadata["license_info"],
        data_is_annotated=True,
        anndata_version=anndata.__version__,
        data_is_minified=minified,
    )
    return HubModel(
        model_path,
        metadata=hub_metadata,
        model_card=model_card,
    )


def load_and_upload_models_for_tissue(base_model_dir: str, adata: AnnData, tissue: str, config: str):
    """Load and upload the models for a given tissue."""
    import os

    from scvi_hub_models.utils import upload_hub_model

    scvi_path = os.path.join(base_model_dir, "scvi")
    if os.path.isdir(scvi_path):
        from scvi.model import SCVI

        scvi_model = SCVI.load(scvi_path, adata=adata)
        mini_scvi_path = minify_and_save_model(scvi_model, adata, base_model_dir)

        scvi_hub_model = create_hub_model(mini_scvi_path, tissue, config, minified=True)
        upload_hub_model(scvi_hub_model, tissue, "scvi")

    scanvi_path = os.path.join(base_model_dir, "scanvi")
    if os.path.isdir(scanvi_path):
        from scvi.model import SCANVI

        scanvi_model = SCANVI.load(scanvi_path, adata=adata)
        mini_scanvi_path = minify_and_save_model(scanvi_model, adata, base_model_dir)

        scanvi_hub_model = create_hub_model(mini_scanvi_path, tissue, config, minified=True)
        upload_hub_model(scanvi_hub_model, tissue, "scanvi")

    condscvi_path = os.path.join(base_model_dir, "condscvi")
    if os.path.isdir(condscvi_path):
        from scvi.model import CondSCVI

        condscvi_model = CondSCVI.load(condscvi_path, adata=adata)
        condscvi_model.save(condscvi_path, overwrite=True, save_anndata=True)

        condscvi_hub_model = create_hub_model(condscvi_path, tissue, config, minified=False)
        upload_hub_model(condscvi_hub_model, tissue, "condscvi")

    stereoscope_path = os.path.join(base_model_dir, "stereoscope")
    if os.path.isdir(stereoscope_path):
        from scvi.external import RNAStereoscope

        stereoscope_model = RNAStereoscope.load(stereoscope_path, adata=adata)
        stereoscope_model.save(stereoscope_path, overwrite=True, save_anndata=True)

        stereoscope_hub_model = create_hub_model(stereoscope_path, tissue, config, minified=False)
        upload_hub_model(stereoscope_hub_model, tissue, "stereoscope")

    del scvi_hub_model
    del scanvi_hub_model
    del condscvi_hub_model
    del stereoscope_hub_model


def load_and_upload_models(base_model_dirs: dict, adata_paths: dict, config: dict):
    """Docstring"""
    from anndata import read_h5ad

    for tissue in config["tissues"]:
        adata = read_h5ad(adata_paths[tissue])
        load_and_upload_models_for_tissue(base_model_dirs[tissue], adata, tissue, config)
        del adata


def model_workflow():
    """Run the model workflow."""
    import json

    with open("../../config/tabula_sapiens.json") as f:
        config = json.load(f)
    # save_dir = TemporaryDirectory().name
    save_dir = "./data"

    # download models and datasets
    base_model_dirs = download_models(config, save_dir)
    adata_paths = download_adatas(config, save_dir)

    # # load and upload models
    load_and_upload_models(base_model_dirs, adata_paths, config)
