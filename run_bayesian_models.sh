#!/bin/bash

#SBATCH --job-name=bayes_all
#SBATCH --output=logs/bayes_all_%A_%a.out
#SBATCH --error=logs/bayes_all_%A_%a.err
#SBATCH --array=0-14
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=05:00:00

set -e

PROJECT_DIR="$HOME/Documents/Activismo/Electometro/bayesian_model_decide2026"

cd "$PROJECT_DIR"

source venv/bin/activate

mkdir -p logs

# ----------------------------------------------------------------------
# Model / feature configuration
# ----------------------------------------------------------------------

MODELS=(
    "gender"
    "gender"
    "gender"

    "region"
    "region"
    "region"

    "education"
    "education"
    "education"

    "age_linear"
    "age_linear"
    "age_linear"

    "age_nonlinear"
    "age_nonlinear"
    "age_nonlinear"
)

FEATURES=(
    "position_importance"
    "combined"
    "all"

    "position_importance"
    "combined"
    "all"

    "position_importance"
    "combined"
    "all"

    "position_importance"
    "combined"
    "all"

    "position_importance"
    "combined"
    "all"
)

TARGET_FILES=(
    "gender_test.csv"
    "gender_test.csv"
    "gender_test.csv"

    "region_test.csv"
    "region_test.csv"
    "region_test.csv"

    "education_test.csv"
    "education_test.csv"
    "education_test.csv"

    "age_test.csv"
    "age_test.csv"
    "age_test.csv"

    "age_test.csv"
    "age_test.csv"
    "age_test.csv"
)

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
FEATURE="${FEATURES[$SLURM_ARRAY_TASK_ID]}"
TARGET_FILE="${TARGET_FILES[$SLURM_ARRAY_TASK_ID]}"

MODEL_DIR="results/bayesian_models/${MODEL}_${FEATURE}"

MODEL_FILE="${MODEL_DIR}/${MODEL}_${FEATURE}.nc"

OUTPUT_DIR="results/validation/${MODEL}_${FEATURE}"

TEST_FILE="data/validation_splits/${TARGET_FILE}"


echo "======================================================================"
echo "Bayesian model + validation"
echo "======================================================================"

echo "Job ID:       $SLURM_JOB_ID"
echo "Array ID:     $SLURM_ARRAY_TASK_ID"
echo "Node:         $SLURM_JOB_NODELIST"
echo "Model:        $MODEL"
echo "Features:     $FEATURE"
echo "Test file:    $TEST_FILE"
echo "Model file:   $MODEL_FILE"
echo "Output dir:   $OUTPUT_DIR"

echo "======================================================================"


# ----------------------------------------------------------------------
# Check test data
# ----------------------------------------------------------------------

if [ ! -f "$TEST_FILE" ]; then
    echo "ERROR: Test file not found:"
    echo "  $TEST_FILE"
    exit 1
fi


# ----------------------------------------------------------------------
# 1. Fit Bayesian model
# ----------------------------------------------------------------------

echo ""
echo "======================================================================"
echo "STEP 1: Fitting Bayesian model"
echo "======================================================================"

python3 src/04_bayesian_models.py \
    --model "$MODEL" \
    --features "$FEATURE" \
    --input data/clean.csv \
    --output-dir "$MODEL_DIR" \
    --random-seed 42


# ----------------------------------------------------------------------
# Check that posterior was created
# ----------------------------------------------------------------------

if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: Bayesian model did not produce expected posterior:"
    echo "  $MODEL_FILE"
    exit 1
fi


# ----------------------------------------------------------------------
# 2. Test Bayesian model on held-out data
# ----------------------------------------------------------------------

echo ""
echo "======================================================================"
echo "STEP 2: Running held-out test"
echo "======================================================================"

python3 src/06_sensitivity_validation_predictions.py \
    "$TEST_FILE" \
    --model "$MODEL" \
    --features "$FEATURE" \
    --model-file "$MODEL_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --random-seed 42


# ----------------------------------------------------------------------
# Finished
# ----------------------------------------------------------------------

echo ""
echo "======================================================================"
echo "Model + validation completed successfully"
echo "======================================================================"

echo "Model:        $MODEL"
echo "Features:     $FEATURE"
echo "Posterior:    $MODEL_FILE"
echo "Validation:   $OUTPUT_DIR"

echo "======================================================================"
