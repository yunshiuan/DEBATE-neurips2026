PYTHON ?= python
DATA_ROOT ?= data
OUTPUT_DIR ?= outputs/paper

.PHONY: download verify reproduce test
download:
	$(PYTHON) -m debate download --data-root "$(DATA_ROOT)"

verify:
	$(PYTHON) -m debate verify --data-root "$(DATA_ROOT)"

reproduce:
	$(PYTHON) -m debate reproduce all --data-root "$(DATA_ROOT)" --output-dir "$(OUTPUT_DIR)"

test:
	$(PYTHON) -m pytest -q
