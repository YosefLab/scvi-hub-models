"""Download source H5ADs published with CELLxGENE Census, without depending on ``cellxgene-census``.

Replaces ``cellxgene_census.download_source_h5ad``. That function opens the Census SOMA store
(which requires ``tiledbsoma``) only to look up the H5AD path for a dataset. The Census release
directory is public JSON and the H5ADs sit in a public S3 bucket, so both can be read over HTTPS.
"""

import json
import os
import urllib.request

CENSUS_RELEASE_DIRECTORY_URL = "https://census.cellxgene.cziscience.com/cellxgene-census/v1/release.json"


def _resolve_h5ads_base_url(census_version: str) -> str:
    """Return the HTTPS base URL of the H5AD folder for a Census version or alias (e.g. ``"stable"``)."""
    with urllib.request.urlopen(CENSUS_RELEASE_DIRECTORY_URL, timeout=60) as response:
        directory = json.load(response)

    description = directory.get(census_version)
    # Aliases such as "stable" and "latest" point to a release name; follow them.
    while isinstance(description, str):
        description = directory.get(description)
    if description is None:
        raise KeyError(f"Unknown Census version: {census_version!r}")

    h5ads = description["h5ads"]
    bucket_and_key = h5ads["uri"].removeprefix("s3://")
    bucket, _, prefix = bucket_and_key.partition("/")
    region = h5ads.get("s3_region", "us-west-2")
    return f"https://{bucket}.s3.{region}.amazonaws.com/{prefix}"


def download_source_h5ad(dataset_id: str, to_path: str, *, census_version: str = "stable") -> str:
    """Download the source H5AD of a CELLxGENE dataset as published in a Census release.

    Parameters
    ----------
    dataset_id
        CELLxGENE ``dataset_id``.
    to_path
        Local file path to write the H5AD to.
    census_version
        Census release name or alias. Defaults to ``"stable"``, matching ``cellxgene_census``.

    Returns
    -------
    The local path of the downloaded file.
    """
    from pooch import retrieve

    url = _resolve_h5ads_base_url(census_version).rstrip("/") + f"/{dataset_id}.h5ad"
    to_path = os.path.abspath(to_path)
    return retrieve(
        url=url,
        known_hash=None,
        fname=os.path.basename(to_path),
        path=os.path.dirname(to_path),
        progressbar=True,
    )
