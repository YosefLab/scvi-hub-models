import logging

#TODO: Remove after retraining models, incompatibility pandas versions
import sys

import pandas

sys.modules["pandas.core.indexes.numeric"] = pandas.core.indexes.base
pandas.core.indexes.base.Int64Index = pandas.core.indexes.base.Index

from scvi_hub_models.models import BaseModelWorkflow

logger = logging.getLogger(__name__)


class _Workflow(BaseModelWorkflow):

    def get_model_collection(self, tissue, base_model_url):
        """Download the models for a given tissue from Zenodo.

        Returns the path to the directory containing the models.
        """
        logging.info(f"Downloading models for {tissue}.")
        if self.dry_run:
            return ""
        from pathlib import Path

        from pooch import Untar, retrieve

        untarred = retrieve(
            url=base_model_url["links"]["self"],
            known_hash=base_model_url["checksum"],
            fname=f"{tissue}_models",
            path=self.save_dir,
            processor=Untar(),
        )
        untarred = sorted(untarred)
        return str(Path(untarred[-1]).parent.parent)

    def get_download_links(self):
        """Download the datasets for a list of tissues from Zenodo."""
        import requests
        res = requests.get(self.config["extra_data_kwargs"]["zenodo_url"])
        files = res.json()['files']

        adata_urls = {}
        base_model_urls = {}
        for file in files:
            tissue = '_'.join(file['key'].split('_')[: -2])
            if tissue not in self.config["extra_data_kwargs"]["tissues"]:
                continue
            if file['key'].endswith(self.config["extra_data_kwargs"]["adata_suffix"]):
                adata_urls[tissue] = file
            elif file['key'].endswith(self.config["extra_data_kwargs"]["models_suffix"]):
                base_model_urls[tissue] = file
            else:
                continue

        return adata_urls, base_model_urls

    def _copy_tensorboard_logs(self, model_dir, model_path):
        """Copy tensorboard logs from model_dir to model_path."""
        import os
        import shutil

        tensorboard_logs = os.path.join(model_dir, "tensorboard_logs")
        if os.path.exists(tensorboard_logs):
            shutil.copytree(tensorboard_logs, os.path.join(model_path, "tensorboard_logs"))

    def run(self):
        super().run()

        adata_urls, base_model_urls = self.get_download_links()

        for tissue in self.config["extra_data_kwargs"]["tissues"]:
            adata_url, base_model_url = adata_urls[tissue], base_model_urls[tissue]
            adata = self._get_adata(
                adata_url["links"]["self"],
                adata_url["checksum"],
                f"{tissue}_adata.h5ad"
            )
            model_collection_dir = self.get_model_collection(
                tissue,
                base_model_url
            )
            for model_name in self.config["extra_data_kwargs"]["models"]:
                logging.info(f"Processing currently model: {tissue} {model_name}.")
                import os
                model_dir = os.path.join(model_collection_dir, model_name.lower())
                model = self._load_model(model_dir, adata, model_name)
                model_path = self._minify_and_save_model(model, adata)
                self._copy_tensorboard_logs(model_dir, model_path)
                hub_model = self._create_hub_model(model_path)
                hub_model = self._upload_hub_model(
                    hub_model, repo_name=f"scvi-tools/tabula-sapiens-{tissue.lower()}-{model_name.lower()}")
