# Makefile for the Interpretable Phonocardiogram Classification pipeline.
# Each target runs one phase script; phases are artifact-driven, so a later
# phase reads what the earlier one wrote to disk. Override the interpreter or
# config on the command line, e.g.:  make evaluate PYTHON=python3.11
#
#   make            # full pipeline (== scripts/run_all.sh)
#   make features   # run a single phase
#   make clean      # remove generated artifacts

PYTHON ?= python
CONFIG ?= configs/config.yaml
RUN     = $(PYTHON) scripts

.PHONY: all run \
        data preprocess features cluster classical cnn evaluate \
        shap gradcam alignment report site-shortcut \
        paper clean help

## full pipeline -------------------------------------------------------------
all run:
	bash scripts/run_all.sh --config $(CONFIG)

## individual phases ---------------------------------------------------------
data:          ; $(RUN)/00_download_data.py     --config $(CONFIG)
preprocess:    ; $(RUN)/01_preprocess_audio.py  --config $(CONFIG)
features:      ; $(RUN)/02_extract_features.py  --config $(CONFIG)
cluster:       ; $(RUN)/03_cluster_features.py  --config $(CONFIG)
classical:     ; $(RUN)/04_train_classical.py   --config $(CONFIG)
cnn:           ; $(RUN)/05_train_cnn.py         --config $(CONFIG)
evaluate:      ; $(RUN)/06_evaluate_models.py   --config $(CONFIG)
shap:          ; $(RUN)/07_explain_shap.py      --config $(CONFIG)
gradcam:       ; $(RUN)/08_explain_gradcam.py   --config $(CONFIG)
alignment:     ; $(RUN)/09_cycle_alignment.py   --config $(CONFIG)
report:        ; $(RUN)/10_build_report_assets.py --config $(CONFIG)
site-shortcut: ; $(RUN)/11_site_shortcut.py     --config $(CONFIG)

## report --------------------------------------------------------------------
paper:
	cd paper && pdflatex -interaction=nonstopmode main.tex \
		&& bibtex main && pdflatex -interaction=nonstopmode main.tex \
		&& pdflatex -interaction=nonstopmode main.tex

## housekeeping --------------------------------------------------------------
clean:
	rm -rf data/interim data/processed figures/* reports/phase_* \
	       results/* paper/*.aux paper/*.log paper/*.out paper/*.bbl paper/*.blg

help:
	@echo "targets: all | data preprocess features cluster classical cnn evaluate"
	@echo "         shap gradcam alignment report site-shortcut | paper | clean"
