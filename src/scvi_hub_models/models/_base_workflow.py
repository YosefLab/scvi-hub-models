import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import anndata
from anndata import __version__ as anndata_version
from frozendict import frozendict
from pooch import retrieve
from scvi.criticism import create_criticism_report
from scvi.hub import HubMetadata, HubModel, HubModelCardHelper
from scvi.model.base import BaseModelClass

logger = logging.getLogger(__name__)


class BaseModelWorkflow:
    """Base class for model workflows.

    Parameters
    ----------
    save_dir
        The directory in which to save intermediate workflow results. Defaults to a temporary
        directory. Can only be set once.
    dry_run
        If ``True``, the workflow will only emit logs and not actually run. Can only be set once.
    config
        A :class:`~frozendict.frozendict` containing the configuration for the workflow. Can only
        be set once.
    """

    def __init__(
        self,
        save_dir: str | None = None,
        dry_run: bool = False,
        config: frozendict | None = None
    ):
        self.save_dir = save_dir
        self.dry_run = dry_run
        self.config = config

    @property
    def save_dir(self):
        return self._save_dir

    @save_dir.setter
    def save_dir(self, path: str):
        if hasattr(self, "_save_dir"):
            raise AttributeError("`save_dir` can only be set once.")
        elif path is None:
            path = TemporaryDirectory().name
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._save_dir = path

    @property
    def dry_run(self):
        return self._dry_run

    @dry_run.setter
    def dry_run(self, value: bool):
        if hasattr(self, "_dry_run"):
            raise AttributeError("`dry_run` can only be set once.")
        self._dry_run = value

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value: frozendict):
        if hasattr(self, "_config"):
            raise AttributeError("`config` can only be set once.")
        elif isinstance(value, dict):
            value = frozendict(value)
        self._config = value

    def _get_adata(self, url: str, hash: str, file_path: str) -> str:
        logger.info("Downloading and reading data.")
        if self.dry_run:
            return None

        retrieve(
            url=url,
            known_hash=hash,
            fname=file_path,
            path=self.save_dir,
            processor=None,
        )
        return anndata.read_h5ad(os.path.join(self.save_dir, file_path))

    def _download_model(self, url: str, hash: str, file_path: str) -> str:
        logger.info("Downloading model.")
        if self.dry_run:
            return None

        return retrieve(
            url=url, #
            known_hash=hash, #config["adata_hashes"][tissue],
            fname=file_path, # f"{tissue}_adata.h5ad"
            path=self.save_dir,
            processor=None,
        )

    def _load_model(self, model_path: str, adata: anndata.AnnData, model_name: str):
        """Load the model."""
        logger.info("Loading model.")
        if self.dry_run:
            return None
        if model_name == "SCVI":
            from scvi.model import SCVI
            model_cls = SCVI
        elif model_name == "SCANVI":
            from scvi.model import SCANVI
            model_cls = SCANVI
        elif model_name == "CondSCVI":
            from scvi.model import CondSCVI
            model_cls = CondSCVI
        elif model_name == "RNAStereoscope":
            from scvi.external import RNAStereoscope
            model_cls = RNAStereoscope
        else:
            raise ValueError(f"Model {model_name} not recognized.")

        model = model_cls.load(os.path.join(self.save_dir, model_path), adata=adata)
        return model

    def _create_hub_model(
            self,
            model_path: str,
            training_data_url: str | None = None
        ) -> HubModel | None:
        logger.info("Creating the HubModel.")
        if self.dry_run:
            return None

        if training_data_url is None:
            training_data_url = self.config.get("training_data_url", None)

        metadata = self.config["metadata"]
        hub_metadata = HubMetadata.from_dir(
            model_path,
            anndata_version=anndata_version
        )
        model_card = HubModelCardHelper.from_dir(
            model_path,
            anndata_version=anndata_version,
            license_info=metadata.get("license_info", "mit"),
            data_modalities=metadata.get("data_modalities", None),
            tissues=metadata.get("tissues", None),
            data_is_annotated=metadata.get("data_is_annotated", False),
            data_is_minified=metadata.get("data_is_minified", False),
            training_data_url=training_data_url,
            training_code_url=metadata.get("training_code_url", None),
            description=metadata.get("description", None),
            references=metadata.get("references", None),
        )

        return HubModel(model_path, hub_metadata, model_card)

    def _minify_and_save_model(
            self,
            model: BaseModelClass,
            adata: anndata.AnnData,
        ) -> str:
        logger.info("Creating the HubModel and creating criticism report.")
        if self.dry_run:
            return None

        if self.config.get("minify_model", True):
            model_name = model.__class__.__name__
            mini_model_path = os.path.join(self.save_dir, f"mini_{model_name.lower()}")
        else:
            mini_model_path = os.path.join(self.save_dir, model.__class__.__name__.lower())

        if not os.path.exists(mini_model_path):
            os.makedirs(mini_model_path)
        if self.config.get("create_criticism_report", True):
            create_criticism_report(
                model,
                save_folder=mini_model_path,
                n_samples=self.config["criticism_settings"].get("n_samples", 3),
                label_key=self.config["criticism_settings"].get("cell_type_key", None)
            )

        if self.config.get("minify_model", True):
            qzm_key = f"{model_name.lower()}_latent_qzm"
            qzv_key = f"{model_name.lower()}_latent_qzv"
            if qzm_key not in adata.obsm and qzv_key not in adata.obsm:
                qzm, qzv = model.get_latent_representation(give_mean=False, return_dist=True)
                adata.obsm[qzm_key] = qzm
                adata.obsm[qzv_key] = qzv
            model.minify_adata(use_latent_qzm_key=qzm_key, use_latent_qzv_key=qzv_key)
        model.save(mini_model_path, overwrite=True, save_anndata=True)

        print(model.adata)

        return mini_model_path

    def _upload_hub_model(self, hub_model: HubModel, repo_name: str | None = None, **kwargs) -> HubModel:
        """Upload the HubModel to Hugging Face."""
        collection_name = self.config.get("collection_name", None)
        if repo_name is None:
            repo_name = self.repo_name
        print('TTTTTT', repo_name)
        logger.info(f"Uploading the HubModel to {repo_name}. Collection: {collection_name}.")

        if not self.dry_run:
            hub_model.push_to_huggingface_hub(
                repo_name=repo_name,
                repo_token=os.environ.get("HF_API_TOKEN", None),
                repo_create=True,
                repo_create_kwargs={"exist_ok": True},
                collection_name=collection_name,
                **kwargs
            )
        return hub_model

    @property
    def id(self):
        return "base-workflow"

    @property
    def repo_name(self) -> str:
        return self.config.get("repo_name", f"scvi-tools/{self.id}")

    def __repr__(self) -> str:
        return f"{self.id} with dry_run={self.dry_run} and save_dir={self.save_dir}."

    def run(self):
        """Run the workflow."""
        logger.info(f"Running {self}.")
