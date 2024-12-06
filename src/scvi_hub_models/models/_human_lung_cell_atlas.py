import logging
import os

import anndata

from scvi_hub_models.models import BaseModelWorkflow

logger = logging.getLogger(__name__)


class _Workflow(BaseModelWorkflow):

    def _download_model(self):
        from pathlib import Path

        from pooch import Unzip, retrieve

        untarred = retrieve(
            url=self.config['extra_data_kwargs']["legacy_model_url"],
            known_hash=self.config['extra_data_kwargs']["legacy_model_hash"],
            fname=self.config['extra_data_kwargs']["legacy_model_dir"],
            processor=Unzip(),
            path=self.save_dir,
        )
        untarred = sorted(untarred)
        print(untarred)
        return str(Path(untarred[0]).parent)

    def _get_model(self) -> str:
        """Download and convert the legacy scANVI model."""
        logging.info("Downloading model.")
        if self.dry_run:
            return None
        from scvi.model import SCANVI

        model_path = os.path.join(self.save_dir, self.config["model_dir"])
        legacy_model_path = self._download_model()
        SCANVI.convert_legacy_save(legacy_model_path, model_path, overwrite=True)

        return model_path

    def _download_reference_adata(self) -> anndata.AnnData:
        """Download the reference (core) dataset from CxG."""
        from cellxgene_census import download_source_h5ad

        adata_path = os.path.join(self.save_dir, self.config['extra_data_kwargs']["reference_adata_fname"])
        if not os.path.exists(adata_path):
            download_source_h5ad(self.config['extra_data_kwargs']["reference_adata_cxg_id"], to_path=adata_path)

        ref_adata = anndata.io.read_h5ad(adata_path)

        return ref_adata

    def _preprocess_reference_adata(self, adata: anndata.AnnData, model_path: str) -> anndata.AnnData:
        """Preprocess the reference dataset.

        1. Set .X to raw counts
        2. Subset to genes that the model was trained on
        3. Remove unnecessary .var columns
        4. Pad empty genes with zeros
        """
        from scvi.model.base import ArchesMixin
        from scvi.model.base._save_load import _load_saved_files

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

    def _postprocess_reference_adata(self, adata: anndata.AnnData) -> anndata.AnnData:
        """Postprocess the reference dataset by adding feature names for padded genes."""
        gene_ids = [
            "ENSG00000253701",
            "ENSG00000269936",
            "ENSG00000274961",
            "ENSG00000279576",
        ]
        feat_names = ["AL928768.3", "RP11-394O4.5", "RP3-492J12.2", "AP000769.1"]
        adata.var["feature_name"] = adata.var["feature_name"].cat.add_categories(feat_names)
        for gene, feature in zip(gene_ids, feat_names):  # noqa: B905
            adata.var.loc[gene, "feature_name"] = feature

        return adata

    def _download_embedding_adata(self) -> str:
        """Download the embedding dataset from Zenodo.

        Embedding dataset contains precomputed latent representations for core cells.
        """
        from pooch import retrieve

        adata = retrieve(
            url=self.config['extra_data_kwargs']["embedding_adata_url"],
            known_hash=self.config['extra_data_kwargs']["embedding_adata_hash"],
            fname=self.config['extra_data_kwargs']["embedding_adata_fname"],
            processor=None,
            path=self.save_dir,
        )
        adata = anndata.io.read_h5ad(adata)
        return adata[adata.obs["core_or_extension"] == "core"].copy()

    def _get_adata(self, model_path: str) -> anndata.AnnData:
        logging.info("Loading data.")
        if self.dry_run:
            return None
        ref_adata = self._download_reference_adata()
        ref_adata = self._preprocess_reference_adata(ref_adata, model_path)
        ref_adata = self._postprocess_reference_adata(ref_adata)
        return ref_adata

    @property
    def id(self) -> str:
        return "human-lung-cell-atlas-scanvi"

    def run(self):
        super().run()

        model_path = self._get_model()
        adata = self._get_adata(model_path)
        model = self._load_model(model_path, adata, "SCANVI")
        model_path = self._minify_and_save_model(model, adata)
        hub_model = self._create_hub_model(model_path)
        hub_model = self._upload_hub_model(hub_model)
