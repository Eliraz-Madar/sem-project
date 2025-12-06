# sem-project

Utilities for training the MCC (argument component) classifier and related document classification experiments.

## Data preparation
- The repository includes the Webis editorials corpus under `src/corpus-webis-editorials-16/annotated-txt/split-by-portal-final/`.
- Convert it to MCC-ready JSONL with:
	```bash
	cd src
	python -m scripts.convert_webis_to_jsonl
	```
	This writes `src/data/editorials.jsonl` (UTF-8, keys: `sentence`, `label`). Labels are normalized to `claim/premise/other`. Output is deterministic; reruns overwrite the file.
- Quick smoke test (runs conversion, reports counts, shows first lines):
	```bash
	cd src
	python -m scripts.test
	```

## Training the MCC model
- Default training script expects `data/mcc_train.jsonl` and `data/mcc_dev.jsonl` relative to `src/`. You can point to `editorials.jsonl` instead:
	```bash
	cd src
	python -m scripts.run_mcc_training --train data/editorials.jsonl --dev data/editorials.jsonl
	```
- To use the default filenames, copy or symlink:
	```bash
	cd src
	copy data\editorials.jsonl data\mcc_train.jsonl
	copy data\editorials.jsonl data\mcc_dev.jsonl
	```

## Notes
- Requires Python 3.10+ and the dependencies listed in `requirements` (not bundled here); HuggingFace `transformers` and `torch` are expected to be installed.
- All scripts read/write UTF-8.