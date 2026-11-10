import urllib.request

import pytest

from scvi_hub_models.models._cxg import _resolve_h5ads_base_url


@pytest.mark.parametrize("census_version", ["stable", "latest"])
def test_resolve_h5ads_base_url(census_version):
    base_url = _resolve_h5ads_base_url(census_version)
    assert base_url.startswith("https://")
    assert base_url.endswith("/h5ads/")


def test_resolve_h5ads_base_url_unknown_version():
    with pytest.raises(KeyError):
        _resolve_h5ads_base_url("not-a-census-version")


def test_source_h5ad_is_reachable():
    # mouse_thymus_cite reference dataset; HEAD request only, nothing is downloaded.
    url = _resolve_h5ads_base_url("stable") + "c14c54f8-85d8-45db-9de7-6ab572cc748a.h5ad"
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        assert response.status == 200
