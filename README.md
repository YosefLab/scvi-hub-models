# scvi-hub-models

This repository contains code to upload models to HuggingFace. It is intended to be rerun with new releases to yield compatible HuggingFace models.
In config, you can find json files that define custom parameters that are stored in HuggingFace and used to create the upload files.
In models, we provide routines for uploading the files.

The usage pattern is to call `python src/scvi_hub_models/ --model_name "MODEL"` with model being one of the file names in config. You can run dry_run
to only execute the procedure without any real execution and can define a save_dir by default we store things in a temporary folder.

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/yoseflab/scvi-hub-models/issues
[changelog]: https://scvi-hub-models.readthedocs.io/latest/changelog.html
