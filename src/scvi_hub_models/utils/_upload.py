from __future__ import annotations

from scvi.hub import HubModel


def upload_hub_model(hub_model: HubModel, repo_name: str, **kwargs) -> HubModel:
    """Upload the HubModel to HuggingFace Hub."""
    import os

    hub_model.push_to_huggingface_hub(repo_name=repo_name, repo_token=os.environ["HF_API_TOKEN"], **kwargs)
    return hub_model
