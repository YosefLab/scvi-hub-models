import logging
import os

import scanpy as sc
from anndata import AnnData
from mudata import MuData
from scvi.model import TOTALVI

from scvi_hub_models.models import BaseModelWorkflow

logger = logging.getLogger(__name__)


class _Workflow(BaseModelWorkflow):

    def _load_adata(self) -> AnnData:
        from cellxgene_census import download_source_h5ad

        adata_path = os.path.join(self.save_dir, self.config['extra_data_kwargs']["reference_adata_fname"])
        if not os.path.exists(adata_path):
            download_source_h5ad(self.config['extra_data_kwargs']["reference_adata_cxg_id"], to_path=adata_path)
        return sc.read_h5ad(adata_path)

    def _preprocess_adata(self, adata: AnnData) -> AnnData:
        import scanpy as sc

        sc.pp.filter_genes(adata, min_counts=3)
        adata.layers["counts"] = adata.X.copy()
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=4000,
            subset=True,
            layer="counts",
            flavor="seurat_v3",
            batch_key="sample_id",
            span=1.0,
        )
        protein_adata = AnnData(
            adata.uns['antibody_raw.X'].toarray(),
            obs=adata.obs,
            var=adata.uns['antibody_features'])

        protein_adata.obs_names = adata.obs_names
        del adata.uns['antibody_raw.X']
        del adata.uns['antibody_features']
        del adata.uns['antibody_X']
        del adata.uns['neighbors']
        mdata = MuData({"rna": adata, "protein": protein_adata})

        return mdata

    def download_adata(self, path) -> AnnData | None:
        """Download and load the dataset."""
        logger.info(f"Saving dataset to {self.save_dir} and preprocessing.")
        if self.dry_run:
            return None
        adata = self._load_adata()
        mdata = self._preprocess_adata(adata)
        mdata.write_h5mu(path)
        return mdata

    def _initialize_model(self, mdata: MuData) -> TOTALVI:
        TOTALVI.setup_mudata(
            mdata,
            rna_layer="counts",
            protein_layer=None,
            batch_key="donor_id",
            modalities={
                "rna_layer": "rna",
                "protein_layer": "protein",
                "batch_key": "rna",
            },
        )
        return TOTALVI(mdata)

    def _train_model(self, model: TOTALVI) -> TOTALVI:
        """Train the scVI model."""
        model.train(max_epochs=50)

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
