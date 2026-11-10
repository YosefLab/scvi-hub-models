# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

-   Basic tool, preprocessing and plotting functions
-   Smoke test (`tests/test_basic.py`) so the CI test job collects at least one test

### Changed

-   Source H5ADs for the Haniffa COVID PBMC, mouse thymus CITE-seq and Human Lung Cell Atlas
    workflows are now downloaded directly over HTTPS from the public CELLxGENE Census bucket
    (`scvi_hub_models.models._cxg.download_source_h5ad`). The files are the same ones
    `cellxgene_census.download_source_h5ad` returned, from the same Census release
-   CI test matrix covers Python 3.12 and 3.14 again

### Removed

-   `cellxgene-census` dependency, which also drops `tiledbsoma`. `tiledbsoma` has no Python 3.14
    wheel, and its source build fails against pybind11 3.x
