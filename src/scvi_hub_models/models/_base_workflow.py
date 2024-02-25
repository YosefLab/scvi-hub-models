import logging
import os
from tempfile import TemporaryDirectory

from frozendict import frozendict
from scvi.hub import HubModel

from scvi_hub_models.utils import make_parents

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
        self, save_dir: str | None = None, dry_run: bool = False, config: frozendict | None = None
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

        make_parents(path)
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

    def create_hub_model(self, model_path: str) -> HubModel | None:
        """Create a HubModel from a model path."""
        from anndata import __version__ as anndata_version
        from scvi.hub import HubMetadata, HubModelCardHelper

        logger.info("Creating the HubModel.")
        if self.dry_run:
            return None

        metadata = self.config["metadata"]
        hub_metadata = HubMetadata.from_dir(model_path, anndata_version=anndata_version)
        model_card = HubModelCardHelper.from_dir(
            model_path,
            anndata_version=anndata_version,
            license_info=metadata.get("license_info", "mit"),
            data_modalities=metadata.get("data_modalities", None),
            tissues=metadata.get("tissues", None),
            data_is_annotated=metadata.get("data_is_annotated", False),
            data_is_minified=metadata.get("data_is_minified", False),
            training_data_url=metadata.get("training_data_urls", None),
            training_code_url=metadata.get("training_code_url", None),
            description=metadata.get("description", None),
            references=metadata.get("references", None),
        )

        return HubModel(model_path, hub_metadata, model_card)

    def upload_hub_model(self, hub_model: HubModel, **kwargs) -> HubModel:
        """Upload the HubModel to Hugging Face."""
        logger.info(f"Uploading the HubModel to {self.repo_name}.")

        if self.dry_run:
            return hub_model

        hub_model.push_to_huggingface_hub(
            repo_name=self.repo_name,
            repo_token=os.environ["HF_API_TOKEN"],
            repo_create_kwargs={"exist_ok": True},
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
