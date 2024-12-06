import logging

from anndata import AnnData
from scvi.model import SCVI

from scvi_hub_models.models import BaseModelWorkflow

logger = logging.getLogger(__name__)


class _Workflow(BaseModelWorkflow):

    def _load_adata(self) -> AnnData:
        from scvi.data import heart_cell_atlas_subsampled

        return heart_cell_atlas_subsampled(save_path=self.save_dir)

    def _preprocess_adata(self, adata: AnnData) -> AnnData:
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

    def load_adata(self) -> AnnData | None:
        """Download and load the dataset."""
        logger.info(f"Saving heart cell atlas dataset to {self.save_dir}.")
        if self.dry_run:
            return None
        adata = self._load_adata()
        adata = self._preprocess_adata(adata)
        return adata

    def _initialize_model(self, adata: AnnData) -> SCVI:
        SCVI.setup_anndata(
            adata,
            layer="counts",
            categorical_covariate_keys=["cell_source", "donor"],
            continuous_covariate_keys=["percent_mito", "percent_ribo"],
        )
        return SCVI(adata)

    def _train_model(self, model: SCVI) -> SCVI:
        """Train the scVI model."""
        model.train(max_epochs=5)

        return model

    def get_model(self, adata) -> SCVI | None:
        """Initialize and train the scVI model."""
        logger.info("Training the scVI model.")
        if self.dry_run:
            return None
        model = self._initialize_model(adata)
        return self._train_model(model)

    def run(self):
        super().run()

        adata = self.load_adata()
        model = self.get_model(adata)
        model_path = self._minify_and_save_model(model, adata)
        hub_model = self._create_hub_model(model_path)
        hub_model = self._upload_hub_model(hub_model)
