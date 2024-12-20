import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import anndata
import git
import mudata
from anndata import __version__ as anndata_version
from dvc.repo import Repo
from frozendict import frozendict
from pooch import retrieve
from scvi.criticism import create_criticism_report
from scvi.hub import HubMetadata, HubModel, HubModelCardHelper
from scvi.model.base import BaseModelClass

# Specify your repository and target file

repo_path = os.path.abspath(Path(__file__).parent.parent.parent.parent)
dvc_repo = Repo(repo_path)
git_repo = git.Repo(repo_path)

logger = logging.getLogger(__name__)

SUPPORTED_PPC_MODELS = [
    "SCVI",
    "SCANVI",
    "CondSCVI",
    "TOTALVI",
]

SUPPORTED_MINIFIED_MODELS = [
    "SCVI",
    "SCANVI",
    "TOTALVI",
]

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
    reload_data
        If ``True``, the data will be reloaded. Otherwise, it will be pulled from DVC. Defaults to ``False``.
    reload_model
        If ``True``, the model will be reloaded. Otherwise, it will be pulled from DVC. Defaults to ``False``.
    """

    def __init__(
        self,
        save_dir: str | None = None,
        dry_run: bool = False,
        config: frozendict | None = None,
        reload_data: bool = True,
        reload_model: bool = True,
    ):
        self.save_dir = save_dir
        self.dry_run = dry_run
        self.config = config
        self.reload_data = reload_data
        self.reload_model = reload_model

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

    @property
    def reload_data(self):
        return self._reload_data

    @reload_data.setter
    def reload_data(self, value: bool):
        if hasattr(self, "_reload_data"):
            raise AttributeError("`reload_data` can only be set once.")
        self._reload_data = value

    @property
    def reload_model(self):
        return self._reload_model

    @reload_model.setter
    def reload_model(self, value: bool):
        if hasattr(self, "_reload_model"):
            raise AttributeError("`reload_model` can only be set once.")
        self._reload_model = value

    def get_adata(self) -> anndata.AnnData | None:
        """Download and load the dataset."""
        logger.info("Loading dataset.")
        if self.dry_run:
            return None
        if self.reload_data:
            path_file = os.path.join(f'{repo_path}/data/', self.config['extra_data_kwargs']['large_training_file_name'])
            print(path_file)
            adata = self.download_adata(path_file)
            dvc_repo.add(path_file)
            git_repo.index.commit(f"Track {path_file} with DVC")
            dvc_repo.push()
            git_repo.remote().push()
        else:
            path_file = os.path.join(f'{repo_path}/data/', self.config['extra_data_kwargs']['large_training_file_name'])
            dvc_repo.pull([path_file])
            if path_file.endswith(".h5mu"):
                adata = mudata.read_h5mu(path_file)
            else:
                adata = anndata.read_h5ad(path_file)
        return adata

    def get_model(self, adata) -> BaseModelClass | None:
        """Download and load the model."""
        logger.info("Loading model.")
        if self.dry_run:
            return None
        if self.reload_model:
            path_file = os.path.join(f'{repo_path}/data/', self.config['model_dir'])
            model = self.load_model(adata)
            model.save(path_file, overwrite=True, save_anndata=False)
            dvc_repo.add(path_file)
            git_repo.index.commit(f"Track {path_file} with DVC")
            dvc_repo.push()
            git_repo.remote().push()
        else:
            path_file = os.path.join(f'{repo_path}/data/', self.config['model_dir'])
            dvc_repo.pull([path_file])
            model = self.default_load_model(adata, self.config['model_class'], path_file)
        return model

    def _get_adata(self, url: str, hash: str, file_path: str, processor: str | None = None) -> str:
        logger.info("Downloading and reading data.")
        if self.dry_run:
            return None

        file_out = retrieve(
            url=url,
            known_hash=hash,
            fname=file_path,
            path=self.save_dir,
            processor=processor,
        )
        return anndata.read_h5ad(file_out)

    def default_load_model(self, adata: anndata.AnnData, model_name: str, model_path: str | None = None) -> BaseModelClass:
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
        elif model_name == "TOTALVI":
            from scvi.model import TOTALVI
            model_cls = TOTALVI
        elif model_name == "Stereoscope":
            from scvi.external import RNAStereoscope
            model_cls = RNAStereoscope
        else:
            raise ValueError(f"Model {model_name} not recognized.")

        if model_path is None:
            model_path = os.path.join(self.save_dir, self.config["model_dir"])
        model = model_cls.load(model_path, adata=adata)
        return model

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
        if self.config.get("create_criticism_report", True) and model.__class__.__name__ in SUPPORTED_PPC_MODELS:
            create_criticism_report(
                model,
                save_folder=mini_model_path,
                n_samples=self.config["criticism_settings"].get("n_samples", 3),
                label_key=self.config["criticism_settings"].get("cell_type_key", None)
            )

        if self.config.get("minify_model", True) and model.__class__.__name__ in SUPPORTED_MINIFIED_MODELS:
            qzm_key = f"{model_name.lower()}_latent_qzm"
            qzv_key = f"{model_name.lower()}_latent_qzv"
            if qzm_key not in adata.obsm and qzv_key not in adata.obsm:
                qzm, qzv = model.get_latent_representation(give_mean=False, return_dist=True)
                adata.obsm[qzm_key] = qzm
                adata.obsm[qzv_key] = qzv
                if isinstance(adata, mudata.MuData):
                    model.minify_mudata(use_latent_qzm_key=qzm_key, use_latent_qzv_key=qzv_key)
                else:
                    model.minify_adata(use_latent_qzm_key=qzm_key, use_latent_qzv_key=qzv_key)
        model.save(mini_model_path, overwrite=True, save_anndata=True)

        return mini_model_path

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

    def _upload_hub_model(self, hub_model: HubModel, repo_name: str | None = None, **kwargs) -> HubModel:
        """Upload the HubModel to Hugging Face."""
        collection_name = self.config.get("collection_name", None)
        if repo_name is None:
            repo_name = self.repo_name
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
