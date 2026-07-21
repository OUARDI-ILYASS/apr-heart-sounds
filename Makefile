# =============================================================================
# Convenience targets. Everything here just calls the phase scripts, so nothing
# is hidden - `make features` and running 02_extract_features.py by hand do
# exactly the same thing.
# =============================================================================

PYTHON ?= python
CONFIG ?= configs/config.yaml

.PHONY: help install test data preprocess features cluster classical cnn \
        evaluate shap gradcam alignment ablations report all paper clean \
        clean-results clean-all

help:
	@echo "APR heart-sound project"
	@echo ""
	@echo "  make install     install Python dependencies"
	@echo "  make test        run the unit tests"
	@echo ""
	@echo "  make data        phase 00  download + census"
	@echo "  make preprocess  phase 01  filter, split, segment"
	@echo "  make features    phase 02  MFCC / log-Mel / PWP"
	@echo "  make cluster     phase 03  k-means + PCA/t-SNE"
	@echo "  make classical   phase 04  SVM + Random Forest"
	@echo "  make cnn         phase 05  CNN training"
	@echo "  make evaluate    phase 06  test-set evaluation"
	@echo "  make shap        phase 07  SHAP explanations"
	@echo "  make gradcam     phase 08  Grad-CAM + sanity checks"
	@echo "  make alignment   phase 09  cardiac-cycle alignment"
	@echo "  make ablations   phase 10  ablation study"
	@echo "  make report      phase 11  dashboard + LaTeX tables"
	@echo ""
	@echo "  make all         phases 00-11 in order"
	@echo "  make paper       build the PDF (needs IEEEtran.cls)"
	@echo "  make clean       remove build products"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v

test-fast:
	$(PYTHON) -m pytest tests/ -q -m "not requires_optional"

data:
	$(PYTHON) scripts/00_download_data.py --config $(CONFIG)
preprocess:
	$(PYTHON) scripts/01_preprocess_audio.py --config $(CONFIG)
features:
	$(PYTHON) scripts/02_extract_features.py --config $(CONFIG)
cluster:
	$(PYTHON) scripts/03_cluster_features.py --config $(CONFIG)
classical:
	$(PYTHON) scripts/04_train_classical.py --config $(CONFIG)
cnn:
	$(PYTHON) scripts/05_train_cnn.py --config $(CONFIG)
evaluate:
	$(PYTHON) scripts/06_evaluate_models.py --config $(CONFIG)
shap:
	$(PYTHON) scripts/07_explain_shap.py --config $(CONFIG)
gradcam:
	$(PYTHON) scripts/08_explain_gradcam.py --config $(CONFIG)
alignment:
	$(PYTHON) scripts/09_cycle_alignment.py --config $(CONFIG)
ablations:
	$(PYTHON) scripts/10_run_ablations.py --config $(CONFIG)
report:
	$(PYTHON) scripts/11_build_report_assets.py --config $(CONFIG)

all:
	bash scripts/run_all.sh

paper:
	cd paper && pdflatex -interaction=nonstopmode main.tex && \
	  bibtex main; pdflatex -interaction=nonstopmode main.tex && \
	  pdflatex -interaction=nonstopmode main.tex

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -f paper/*.aux paper/*.bbl paper/*.blg paper/*.log paper/*.out

clean-results:
	rm -rf results/* figures/* reports/*

clean-all: clean clean-results
	rm -rf data/interim/* data/processed/* models/classical/* models/cnn/* models/scalers/*
