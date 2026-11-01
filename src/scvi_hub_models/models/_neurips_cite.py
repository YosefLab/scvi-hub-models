import logging

import scanpy as sc
from anndata import AnnData
from mudata import MuData
from pooch import Decompress
from scvi.model import TOTALVI

from scvi_hub_models.models import BaseModelWorkflow

logger = logging.getLogger(__name__)


class _Workflow(BaseModelWorkflow):
    def _preprocess_adata(self, adata: AnnData) -> AnnData:
        rna = adata[:, adata.var['feature_types']=='GEX'].copy()
        protein = adata[:, adata.var['feature_types']=='ADT'].copy()
        protein.layers["counts"] = protein.layers["counts"].toarray()
        sc.pp.filter_genes(rna, min_counts=3)
        sc.pp.highly_variable_genes(
            rna,
            n_top_genes=4000,
            subset=True,
            layer="counts",
            flavor="seurat_v3",
            batch_key="Site",
            span=1.0,
        )
        adata = MuData({"rna": rna, "protein": protein})

        return adata

    def download_adata(self, path) -> AnnData | None:
        """Download and load the dataset."""
        logger.info(f"Saving dataset to {path} and preprocessing.")
        if self.dry_run:
            return None
        adata = self._get_adata(
            url=self.config["extra_data_kwargs"]["url"],
            hash=self.config["extra_data_kwargs"]["hash"],
            file_path=self.config["extra_data_kwargs"]["reference_adata_fname"],
            processor=Decompress(),
        )
        mdata = self._preprocess_adata(adata)
        mdata.write_h5mu(path)
        return mdata

    def _initialize_model(self, mdata: MuData) -> TOTALVI:
        TOTALVI.setup_mudata(
            mdata,
            rna_layer="counts",
            protein_layer="counts",
            batch_key="batch",
            modalities={
                "rna_layer": "rna",
                "protein_layer": "protein",
                "batch_key": "rna",
            },
        )
        return TOTALVI(mdata)

    def _train_model(self, model: TOTALVI) -> TOTALVI:
        """Train the scVI model."""
        model.train(max_epochs=200)

        return model

    def load_model(self, adata) -> TOTALVI | None:
        """Initialize and train the scVI model."""
        logger.info("Training the scVI model.")
        if self.dry_run:
            return None
        model = self._initialize_model(adata)
        return self._train_model(model)

    def run(self):
        super().run()

        mdata = self.get_adata()
        model = self.get_model(mdata)
        model_path = self._minify_and_save_model(model, mdata)
        hub_model = self._create_hub_model(model_path)
        hub_model = self._upload_hub_model(hub_model)
