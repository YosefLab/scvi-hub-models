from anndata import AnnData
from scvi.hub import HubModel
from scvi.model import SCANVI


def download_legacy_model(config: dict, save_dir: str) -> str:
    """Download the legacy scANVI model from Zenodo."""
    from pathlib import Path

    from pooch import Unzip, retrieve

    unzipped = retrieve(
        url=config["legacy_model_url"],
        known_hash=config["legacy_model_hash"],
        fname=config["legacy_model_dir"],
        processor=Unzip(),
        path=save_dir,
    )
    unzipped = sorted(unzipped)
    return str(Path(unzipped[0]).parent)


def convert_legacy_model(legacy_model_path: str, config: dict, save_dir: str) -> str:
    """Convert the legacy scANVI model."""
    import os

    from scvi.model import SCANVI

    model_path = os.path.join(save_dir, config["model_dir"])
    SCANVI.convert_legacy_save(legacy_model_path, model_path, overwrite=True)

    return model_path


def download_reference_adata(config: dict, save_dir: str) -> str:
    """Download the reference (core) dataset from CxG."""
    import os

    from cellxgene_census import download_source_h5ad

    adata_path = os.path.join(save_dir, config["reference_adata_fname"])
    if not os.path.exists(adata_path):
        download_source_h5ad(config["reference_adata_cxg_id"], to_path=adata_path)

    return adata_path


def preprocess_reference_adata(adata: AnnData, model_path: str) -> AnnData:
    """Preprocess the reference dataset.

    1. Set .X to raw counts
    2. Subset to genes that the model was trained on
    3. Remove unnecessary .var columns
    4. Pad empty genes with zeros
    """
    from scvi.model.base import ArchesMixin
    from scvi.model.base._utils import _load_saved_files

    # .X does not contain raw counts initially
    adata.X = adata.raw.X
    _, genes, _, _ = _load_saved_files(model_path, load_adata=False)
    adata = adata[:, adata.var.index.isin(genes)].copy()

    # get rid of some var columns that we dont need
    # -- will make later processing easier
    del adata.var["feature_is_filtered"]
    del adata.var["feature_reference"]
    del adata.var["feature_biotype"]

    ArchesMixin.prepare_query_anndata(adata, model_path)

    return adata


def load_model(model_path: str, adata: AnnData) -> SCANVI:
    """Load a scANVI model."""
    return SCANVI.load(model_path, adata=adata)


def postprocess_reference_adata(adata: "AnnData") -> "AnnData":
    """Postprocess the reference dataset by adding feature names for padded genes."""
    gene_ids = [
        "ENSG00000253701",
        "ENSG00000269936",
        "ENSG00000274961",
        "ENSG00000279576",
    ]
    feat_names = ["AL928768.3", "RP11-394O4.5", "RP3-492J12.2", "AP000769.1"]
    adata.var["feature_name"] = adata.var["feature_name"].cat.add_categories(feat_names)
    for gene, feature in zip(gene_ids, feat_names):
        adata.var.loc[gene, "feature_name"] = feature

    return adata


def download_embedding_adata(config: dict, save_dir: str) -> str:
    """Download the embedding dataset from Zenodo.

    Embedding dataset contains precomputed latent representations for core cells.
    """
    from pooch import retrieve

    return retrieve(
        url=config["embedding_adata_url"],
        known_hash=config["embedding_adata_hash"],
        fname=config["embedding_adata_fname"],
        processor=None,
        path=save_dir,
    )


def preprocess_embedding_adata(adata: AnnData) -> AnnData:
    """Preprocess the embedding dataset by subsetting to core cells."""
    return adata[adata.obs["core_or_extension"] == "core"].copy()


def minify_model(model: SCANVI, ref_adata: AnnData, emb_adata: AnnData) -> SCANVI:
    """Minify the model and dataset.

    Uses the precomputed mean latent posterior from the embedding dataset.
    """
    qzm = emb_adata[ref_adata.obs.index].copy().X
    _, qzv = model.get_latent_representation(give_mean=False, return_dist=True)
    qzm_key = "SCANVI_latent_qzm"
    qzv_key = "SCANVI_latent_qzv"
    ref_adata.obsm[qzm_key] = qzm
    ref_adata.obsm[qzv_key] = qzv
    model.minify_adata(use_latent_qzm_key=qzm_key, use_latent_qzv_key=qzv_key)

    return model


def save_minified_model(model: SCANVI, config: dict, save_dir: str) -> str:
    """Save the minified model."""
    import os

    model_path = os.path.join(save_dir, config["mini_model_dir"])
    model.save(model_path, overwrite=True, save_anndata=True)

    return model_path


def create_hub_model(mini_model_path: str, config: dict) -> HubModel:
    """Create a HubModel from the minified model."""
    import anndata
    from scvi.hub import HubMetadata, HubModel, HubModelCardHelper

    metadata = config["metadata"]

    hub_metadata = HubMetadata.from_dir(
        mini_model_path,
        training_data_url=metadata["training_data_url"],
        anndata_version=anndata.__version__,
    )
    model_card = HubModelCardHelper.from_dir(
        mini_model_path,
        training_data_url=metadata["training_data_url"],
        training_code_url=metadata["training_code_url"],
        tissues=metadata["tissues"],
        data_modalities=metadata["data_modalities"],
        description=metadata["description"],
        references=metadata["references"],
        license_info=metadata["license_info"],
        data_is_annotated=metadata["data_is_annotated"],
        anndata_version=anndata.__version__,
        data_is_minified=True,
    )
    return HubModel(
        mini_model_path,
        metadata=hub_metadata,
        model_card=model_card,
    )


def model_workflow(dry_run: bool, repo_create: bool) -> None:
    """Run the model workflow."""
    import json

    from anndata import read_h5ad
    from scvi_hub_model.utils import upload_hub_model

    with open("../../config/human_lung_cell_atlas.json") as f:
        config = json.load(f)
    # save_dir = TemporaryDirectory().name
    save_dir = "./data"

    # download and convert legacy model
    legacy_model_path = download_legacy_model(config, save_dir)
    model_path = convert_legacy_model(legacy_model_path, config, save_dir)

    # download and process reference dataset
    ref_adata_path = download_reference_adata(config, save_dir)
    ref_adata = read_h5ad(ref_adata_path)
    ref_adata = preprocess_reference_adata(ref_adata, model_path)
    model = load_model(model_path, ref_adata)
    ref_adata = postprocess_reference_adata(ref_adata)

    # download and process embedding dataset
    emb_adata_path = download_embedding_adata(config, save_dir)
    emb_adata = read_h5ad(emb_adata_path)
    emb_adata = preprocess_embedding_adata(emb_adata)

    # minify model and save
    model = minify_model(model, ref_adata, emb_adata)
    mini_model_path = save_minified_model(model, config, save_dir)

    # create and upload hub model
    hub_model = create_hub_model(mini_model_path, config)
    hub_model = upload_hub_model(hub_model, "scvi-tools/human-lung-cell-atlas")
