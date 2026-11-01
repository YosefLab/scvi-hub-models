import logging
import os

from anndata import AnnData
from scvi.criticism import create_criticism_report
from scvi.model import SCVI

from scvi_hub_models.models import BaseModelWorkflow

logger = logging.getLogger(__name__)


class _Workflow(BaseModelWorkflow):

    def download_adata(self, path) -> AnnData | None:
        from scvi.data import synthetic_iid

        logger.info("Loading synthetic dataset.")
        if self.dry_run:
            return None
        adata = synthetic_iid()
        adata.write_h5ad(path)
        return adata

    def load_model(self, adata: AnnData) -> SCVI:
        logger.info("Training the scVI model.")
        if self.dry_run:
            return None
        SCVI.setup_anndata(adata)
        model = SCVI(adata)
        model.train(max_epochs=10)
        return model

    @property
    def id(self) -> str:
        return "test-scvi"

    def run(self):
        super().run()

        adata = self.get_adata()
        model = self.get_model(adata)
        model_path = self._minify_and_save_model(model, adata)
        hub_model = self._create_hub_model(model_path)
        hub_model = self._upload_hub_model(hub_model)
