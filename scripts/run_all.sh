#!/usr/bin/env bash
# =============================================================================
# Run the whole pipeline, phase by phase, stopping at the first failure.
#
#   bash scripts/run_all.sh                       # full run
#   bash scripts/run_all.sh --skip-download       # data already on disk
#   bash scripts/run_all.sh --from 04             # resume from phase 04
#   bash scripts/run_all.sh --skip-ablations      # skip the slow phase 10
#
# Each phase is artifact-driven, so resuming is safe: a phase reads what the
# previous one wrote to disk and does not depend on anything held in memory.
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/config.yaml}"

START_FROM="00"
SKIP_DOWNLOAD=""
SKIP_ABLATIONS="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)            START_FROM="$2"; shift 2 ;;
    --skip-download)   SKIP_DOWNLOAD="--skip-download"; shift ;;
    --skip-ablations)  SKIP_ABLATIONS="true"; shift ;;
    --config)          CONFIG="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

PHASES=(
  "00:00_download_data.py:Download and census"
  "01:01_preprocess_audio.py:Preprocess, split, segment"
  "02:02_extract_features.py:Feature extraction"
  "03:03_cluster_features.py:Clustering and projections"
  "04:04_train_classical.py:SVM and Random Forest"
  "05:05_train_cnn.py:CNN training"
  "06:06_evaluate_models.py:Test-set evaluation"
  "07:07_explain_shap.py:SHAP explanations"
  "08:08_explain_gradcam.py:Grad-CAM and sanity checks"
  "09:09_cycle_alignment.py:Cardiac-cycle alignment"
  "10:10_run_ablations.py:Ablation study"
  "11:11_build_report_assets.py:Report assets"
)

echo "=============================================================="
echo " APR heart-sound pipeline"
echo " config : $CONFIG"
echo " from   : phase $START_FROM"
echo "=============================================================="

PIPELINE_START=$(date +%s)

for entry in "${PHASES[@]}"; do
  IFS=':' read -r id script description <<< "$entry"

  if [[ "$id" < "$START_FROM" ]]; then
    echo "  [skip] phase $id — $description"
    continue
  fi
  if [[ "$id" == "10" && "$SKIP_ABLATIONS" == "true" ]]; then
    echo "  [skip] phase 10 — ablations (--skip-ablations)"
    continue
  fi

  echo ""
  echo "--------------------------------------------------------------"
  echo " PHASE $id — $description"
  echo "--------------------------------------------------------------"

  extra=""
  [[ "$id" == "00" ]] && extra="$SKIP_DOWNLOAD"

  PHASE_START=$(date +%s)
  if ! $PYTHON "scripts/$script" --config "$CONFIG" $extra; then
    echo ""
    echo "!! Phase $id FAILED. Pipeline stopped."
    echo "   Log: reports/phase_${script%.py}/run.log"
    exit 1
  fi
  echo "   phase $id finished in $(( $(date +%s) - PHASE_START ))s"
done

echo ""
echo "=============================================================="
echo " Pipeline complete in $(( ($(date +%s) - PIPELINE_START) / 60 )) min"
echo " Dashboard : reports/PIPELINE_STATUS.md"
echo " Figures   : figures/"
echo " Tables    : paper/tables/"
echo "=============================================================="
