import logging
import os

from anndata import AnnData
from scvi.model import SCVI

from scvi_hub_models.models import BaseModelWorkflow

logger = logging.getLogger(__name__)


class _Workflow(BaseModelWorkflow):

    def load_dataset(self) -> AnnData | None:
        from scvi.data import synthetic_iid

        logger.info("Loading synthetic dataset.")
        if self.dry_run:
            return None

        return synthetic_iid()

    def initialize_model(self, adata: AnnData | None) -> SCVI | None:
        logger.info("Initializing the scVI model.")
        if self.dry_run:
            return None

        SCVI.setup_anndata(adata)
        return SCVI(adata)

    def train_model(self, model: SCVI | None) -> SCVI | None:
        logger.info("Training the scVI model.")
        if self.dry_run:
            return model

        model.train(max_epochs=1)
        return model

    def save_model(self, model: SCVI | None) -> str | None:
        logger.info("Saving the scVI model.")

        if self.dry_run:
            return None

        model_path = os.path.join(self.save_dir, self.id)
        model.save(model_path, save_anndata=True, overwrite=True)

        return model_path

    @property
    def id(self) -> str:
        return "test-scvi"

    def run(self):
        super().run()

        adata = self.load_dataset()
        model = self.initialize_model(adata)
        model = self.train_model(model)
        model_path = self.save_model(model)
        hub_model = self.create_hub_model(model_path)
        hub_model = self.upload_hub_model(hub_model)
