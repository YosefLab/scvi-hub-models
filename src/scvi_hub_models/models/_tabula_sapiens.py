import logging
import os
from pathlib import Path

import anndata
import scanpy as sc

from scvi_hub_models.models import BaseModelWorkflow

logger = logging.getLogger(__name__)

# Resolve the repo root the same way _base_workflow does.
_repo_path = os.path.abspath(Path(__file__).parent.parent.parent.parent)


def _print_status(*args):
    """Print a prominent status line (visible alongside logger output)."""
    msg = " ".join(str(a) for a in args)
    print(f"\n{'='*70}\n  {msg}\n{'='*70}", flush=True)


class _Workflow(BaseModelWorkflow):

    # ------------------------------------------------------------------
    # Data loading – bypass the base-class DVC push/pull so we can work
    # with the already-downloaded file without needing dvc-gdrive.
    # ------------------------------------------------------------------

    @staticmethod
    def _is_valid_hdf5(path: str) -> bool:
        """Return True iff *path* exists and is a readable HDF5 file."""
        try:
            import h5py
            with h5py.File(path, "r"):
                return True
        except Exception:
            return False

    def get_adata(self) -> anndata.AnnData | None:
        """Return the full Tabula Sapiens adata, downloading only if absent/corrupt."""
        if self.dry_run:
            return None

        path_file = os.path.join(
            _repo_path, "data",
            self.config["extra_data_kwargs"]["large_training_file_name"],
        )

        if os.path.exists(path_file) and not self._is_valid_hdf5(path_file):
            _print_status(
                f"WARNING: existing file at {path_file} is not a valid HDF5 — re-downloading."
            )
            os.remove(path_file)

        if not os.path.exists(path_file):
            _print_status(f"Downloading Tabula Sapiens dataset → {path_file}")
            self._download_h5ad(path_file)
        else:
            _print_status(f"Using existing dataset at {path_file}")

        _print_status("Reading full h5ad into memory …")
        adata = anndata.read_h5ad(path_file)
        print(f"  Dataset shape: {adata.shape}", flush=True)
        return adata

    def _download_h5ad(self, path: str) -> None:
        """Stream the h5ad directly from CellXGene, using a .tmp file to avoid
        leaving a partial download that would be mistaken for a valid cache entry."""
        import urllib.request

        url = self.config["extra_data_kwargs"]["reference_adata_url"]
        tmp_path = path + ".tmp"
        logger.info(f"Fetching {url} → {tmp_path}")
        try:
            urllib.request.urlretrieve(url, tmp_path)
            os.rename(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    # ------------------------------------------------------------------
    # Per-tissue pre-processing
    # ------------------------------------------------------------------

    def _get_tissue_adata(self, adata: anndata.AnnData, tissue: str) -> anndata.AnnData:
        """Subset adata to one tissue, extract raw counts, and apply HVG filtering."""
        tissue_column = self.config["extra_data_kwargs"].get("tissue_column", "tissue_in_publication")
        batch_key = self.config["extra_data_kwargs"].get("batch_key", "donor_assay")

        tissue_adata = adata[adata.obs[tissue_column] == tissue].copy()
        print(
            f"  Subsetting tissue '{tissue}': "
            f"{tissue_adata.n_obs:,} cells × {tissue_adata.n_vars:,} genes",
            flush=True,
        )

        if tissue_adata.n_obs == 0:
            raise ValueError(
                f"No cells found for tissue '{tissue}'. "
                f"Check '{tissue_column}' column. "
                f"Available values: {sorted(adata.obs[tissue_column].unique())}"
            )

        # Extract raw integer counts from adata.raw into a 'counts' layer.
        # CellXGene h5ad files store normalised floats in X and raw counts in raw.X.
        if tissue_adata.raw is not None:
            raw_var_names = list(tissue_adata.raw.var_names)
            adata_var_names = list(tissue_adata.var_names)
            if raw_var_names == adata_var_names:
                tissue_adata.layers["counts"] = tissue_adata.raw.X.copy()
            else:
                # raw may cover more genes — find matching indices
                idx = [tissue_adata.raw.var_names.get_loc(g) for g in adata_var_names]
                tissue_adata.layers["counts"] = tissue_adata.raw.X[:, idx].copy()
            print("  Extracted raw counts from adata.raw → 'counts' layer.", flush=True)
        else:
            logger.warning("adata.raw is None — using adata.X as counts (may be normalised).")

        print(f"  Selecting top-3000 highly variable genes (batch_key='{batch_key}') …", flush=True)
        hvg_layer = "counts" if "counts" in tissue_adata.layers else None
        sc.pp.highly_variable_genes(
            tissue_adata,
            n_top_genes=3000,
            subset=True,
            flavor="seurat_v3",
            span=1.0,
            batch_key=batch_key,
            layer=hvg_layer,
        )
        print(f"  After HVG: {tissue_adata.n_obs:,} cells × {tissue_adata.n_vars:,} genes", flush=True)
        return tissue_adata

    # ------------------------------------------------------------------
    # Resume helpers
    # ------------------------------------------------------------------

    # Maps user-facing model names to their actual scvi-tools class names, used to
    # match the directory that _minify_and_save_model creates (mini_{classname.lower()}).
    _MODEL_CLASS_NAMES: dict[str, str] = {
        "Stereoscope": "RNAStereoscope",
    }

    def _tissue_model_save_dir(self, tissue: str, model_name: str) -> str:
        """Return a stable, tissue-specific save path for a minified model."""
        class_name = self._MODEL_CLASS_NAMES.get(model_name, model_name)
        return os.path.join(self.save_dir, tissue, f"mini_{class_name.lower()}")

    # ------------------------------------------------------------------
    # Model training helpers
    # ------------------------------------------------------------------

    def _max_epochs(self, default: int) -> int:
        """Return config override if present (for test runs), else the default."""
        return self.config["extra_data_kwargs"].get("max_epochs", default)

    def _train_scvi(self, adata: anndata.AnnData):
        import scvi as scvi_tools

        batch_key = self.config["extra_data_kwargs"].get("batch_key", "donor_assay")
        labels_key = self.config["extra_data_kwargs"].get("labels_key", "cell_type")
        layer = "counts" if "counts" in adata.layers else None
        scvi_model_kwargs = self.config["extra_data_kwargs"].get(
            "scvi_model_kwargs",
            {
                "dropout_rate": 0.05,
                "dispersion": "gene",
                "n_layers": 3,
                "n_latent": 20,
                "gene_likelihood": "nb",
                "use_batch_norm": "none",
                "use_layer_norm": "both",
                "encode_covariates": True,
            },
        )
        epochs = self._max_epochs(100)
        print(f"  [SCVI] Setup (batch='{batch_key}', labels='{labels_key}', layer='{layer}') …", flush=True)
        scvi_tools.model.SCVI.setup_anndata(
            adata, batch_key=batch_key, labels_key=labels_key, layer=layer
        )
        model = scvi_tools.model.SCVI(adata, **scvi_model_kwargs)
        print(f"  [SCVI] Training (max_epochs={epochs}) …", flush=True)
        model.train(max_epochs=epochs, train_size=0.8, plan_kwargs={"n_epochs_kl_warmup": epochs})
        print("  [SCVI] Training complete.", flush=True)
        return model

    def _train_scanvi(self, scvi_model):
        import scvi as scvi_tools

        unknown_celltype_label = self.config["extra_data_kwargs"].get(
            "unknown_celltype_label", "unknown"
        )
        epochs = self._max_epochs(20)
        print("  [SCANVI] Initialising from SCVI model …", flush=True)
        model = scvi_tools.model.SCANVI.from_scvi_model(
            scvi_model, unlabeled_category=unknown_celltype_label
        )
        print(f"  [SCANVI] Training (max_epochs={epochs}) …", flush=True)
        model.train(
            max_epochs=epochs,
            n_samples_per_label=20,
            plan_kwargs={"n_epochs_kl_warmup": min(10, epochs)},
        )
        print("  [SCANVI] Training complete.", flush=True)
        return model

    def _train_condscvi(self, adata: anndata.AnnData):
        import scvi as scvi_tools

        labels_key = self.config["extra_data_kwargs"].get("labels_key", "cell_type")
        layer = "counts" if "counts" in adata.layers else None
        epochs = self._max_epochs(200)
        print(f"  [CondSCVI] Setup (labels='{labels_key}', layer='{layer}') …", flush=True)
        scvi_tools.model.CondSCVI.setup_anndata(adata, labels_key=labels_key, layer=layer)
        model = scvi_tools.model.CondSCVI(
            adata, n_latent=5, n_layers=2, dropout_rate=0.05, weight_obs=False
        )
        print(f"  [CondSCVI] Training (max_epochs={epochs}) …", flush=True)
        model.train(max_epochs=epochs, train_size=0.8)
        print("  [CondSCVI] Training complete.", flush=True)
        return model

    def _train_stereoscope(self, adata: anndata.AnnData):
        import scvi as scvi_tools

        labels_key = self.config["extra_data_kwargs"].get("labels_key", "cell_type")
        layer = "counts" if "counts" in adata.layers else None
        epochs = self._max_epochs(100)
        print(f"  [Stereoscope] Setup (labels='{labels_key}', layer='{layer}') …", flush=True)
        scvi_tools.external.RNAStereoscope.setup_anndata(adata, labels_key=labels_key, layer=layer)
        model = scvi_tools.external.RNAStereoscope(adata)
        print(f"  [Stereoscope] Training (max_epochs={epochs}) …", flush=True)
        model.train(max_epochs=epochs, train_size=0.8)
        print("  [Stereoscope] Training complete.", flush=True)
        return model

    # ------------------------------------------------------------------
    # Main workflow
    # ------------------------------------------------------------------

    def _tissue_raw_scvi_save_dir(self, tissue: str) -> str:
        """Return a stable path for a non-minified (raw) SCVI checkpoint used to init SCANVI."""
        return os.path.join(self.save_dir, tissue, "raw_scvi")

    def run(self):
        super().run()

        if self.save_dir.startswith("/tmp/"):
            logger.warning(
                "save_dir is a temporary directory. Trained models will be lost on restart. "
                "Pass --save_dir /persistent/path to enable resume from saved models."
            )

        tissues = self.config["extra_data_kwargs"]["tissues"]
        models_to_train = self.config["extra_data_kwargs"]["models"]
        n_tissues = len(tissues)

        # Load the full combined dataset once.
        adata = self.get_adata()

        for tissue_idx, tissue in enumerate(tissues, start=1):
            _print_status(
                f"TISSUE {tissue_idx}/{n_tissues}: {tissue}"
                f"  |  models: {models_to_train}"
            )

            tissue_adata = self._get_tissue_adata(adata, tissue)

            needs_scvi = "SCVI" in models_to_train or "SCANVI" in models_to_train
            needs_scanvi = "SCANVI" in models_to_train

            scvi_mini_path = self._tissue_model_save_dir(tissue, "SCVI")
            scanvi_mini_path = self._tissue_model_save_dir(tissue, "SCANVI")
            raw_scvi_path = self._tissue_raw_scvi_save_dir(tissue)

            scvi_model = None
            scanvi_model = None

            if needs_scvi:
                scanvi_needs_training = needs_scanvi and not os.path.exists(scanvi_mini_path)

                if scanvi_needs_training:
                    # SCANVI must be trained from a *non-minified* SCVI model.
                    # Load the raw SCVI checkpoint if available; otherwise train from scratch.
                    if os.path.exists(raw_scvi_path):
                        _print_status(f"  [{tissue}] Loading SCVI raw checkpoint for SCANVI init …")
                        import scvi as scvi_tools
                        scvi_model = scvi_tools.model.SCVI.load(raw_scvi_path, adata=tissue_adata)
                    else:
                        _print_status(f"  [{tissue}] Training SCVI …")
                        scvi_model = self._train_scvi(tissue_adata)
                        # Persist a raw (non-minified) checkpoint so that SCANVI can be
                        # (re-)initialised on resume, even after the minified copy is saved.
                        print(f"  [SCVI] Saving raw checkpoint → {raw_scvi_path}", flush=True)
                        os.makedirs(raw_scvi_path, exist_ok=True)
                        scvi_model.save(raw_scvi_path, overwrite=True, save_anndata=False)

                    # Train SCANVI now — before any minification of the SCVI model.
                    _print_status(f"  [{tissue}] Training SCANVI from non-minified SCVI …")
                    scanvi_model = self._train_scanvi(scvi_model)

                elif not os.path.exists(scvi_mini_path):
                    # Only SCVI needs training (SCANVI is done or not requested).
                    _print_status(f"  [{tissue}] Training SCVI …")
                    scvi_model = self._train_scvi(tissue_adata)

            for model_name in models_to_train:
                mini_path = self._tissue_model_save_dir(tissue, model_name)

                if os.path.exists(mini_path):
                    _print_status(
                        f"  [{tissue}] {model_name} already saved at {mini_path} — uploading only."
                    )
                    hub_model = self._create_hub_model(mini_path)
                    repo_name = f"scvi-tools/tabula-sapiens-{tissue.lower()}-{model_name.lower()}"
                    self._upload_hub_model(hub_model, repo_name=repo_name)
                    print(f"  Upload complete: {repo_name}", flush=True)
                    continue

                _print_status(
                    f"  [{tissue}] Processing {model_name} "
                    f"({models_to_train.index(model_name)+1}/{len(models_to_train)})"
                )

                if model_name == "SCVI":
                    model = scvi_model
                    adata_for_model = tissue_adata
                elif model_name == "SCANVI":
                    # scanvi_model was trained before SCVI was minified (see above).
                    model = scanvi_model
                    adata_for_model = tissue_adata
                elif model_name == "CondSCVI":
                    model = self._train_condscvi(tissue_adata.copy())
                    adata_for_model = model.adata
                elif model_name == "Stereoscope":
                    model = self._train_stereoscope(tissue_adata.copy())
                    adata_for_model = model.adata
                else:
                    raise ValueError(f"Unknown model: {model_name}")

                # Temporarily redirect save_dir to a tissue-specific subdirectory
                # so that each tissue+model lands in its own folder and doesn't
                # overwrite a sibling model.
                print(f"  Minifying and saving {model_name} to {mini_path} …", flush=True)
                original_save_dir = self._save_dir
                self._save_dir = os.path.join(original_save_dir, tissue)
                os.makedirs(self._save_dir, exist_ok=True)
                try:
                    model_path = self._minify_and_save_model(model, adata_for_model)
                finally:
                    self._save_dir = original_save_dir

                repo_name = f"scvi-tools/tabula-sapiens-{tissue.lower()}-{model_name.lower()}"
                print(f"  Uploading to Hugging Face → {repo_name} …", flush=True)
                hub_model = self._create_hub_model(model_path)
                self._upload_hub_model(hub_model, repo_name=repo_name)
                print(f"  Upload complete: {repo_name}", flush=True)

        _print_status("All tissues and models processed successfully.")
