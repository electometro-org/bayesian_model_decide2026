#!/bin/bash

#SBATCH --job-name=06_train_and_test
#SBATCH --output=logs/06_train_and_test_%A_%a.out
#SBATCH --error=logs/06_train_and_test_%A_%a.err
#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=standard

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-14
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=15GB
#SBATCH --time=12:00:00

set -e

PROJECT_DIR="$HOME/bayesian_model_decide2026"

cd "$PROJECT_DIR"

module purge
module add virtualenv/20.32.0-GCCcore-14.3.0
module add Python/3.13.5-GCCcore-14.3.0

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


# ----------------------------------------------------------------------
# Target-specific TRAINING files
# ----------------------------------------------------------------------

TRAIN_FILES=(
    "gender_train.csv"
    "gender_train.csv"
    "gender_train.csv"

    "region_train.csv"
    "region_train.csv"
    "region_train.csv"

    "education_train.csv"
    "education_train.csv"
    "education_train.csv"

    "age_train.csv"
    "age_train.csv"
    "age_train.csv"

    "age_train.csv"
    "age_train.csv"
    "age_train.csv"
)


# ----------------------------------------------------------------------
# Target-specific TEST files
# ----------------------------------------------------------------------

TEST_FILES=(
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


# ----------------------------------------------------------------------
# Select configuration for this array task
# ----------------------------------------------------------------------

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
FEATURE="${FEATURES[$SLURM_ARRAY_TASK_ID]}"

TRAIN_FILE="data/validation_splits/${TRAIN_FILES[$SLURM_ARRAY_TASK_ID]}"
TEST_FILE="data/validation_splits/${TEST_FILES[$SLURM_ARRAY_TASK_ID]}"

MODEL_DIR="results/bayesian_models/${MODEL}_${FEATURE}"

MODEL_FILE="${MODEL_DIR}/${MODEL}_${FEATURE}.nc"

OUTPUT_DIR="results/validation/${MODEL}_${FEATURE}"


# ----------------------------------------------------------------------
# Information
# ----------------------------------------------------------------------

echo "======================================================================"
echo "Bayesian model + held-out validation"
echo "======================================================================"

echo "Job ID:       $SLURM_JOB_ID"
echo "Array ID:     $SLURM_ARRAY_TASK_ID"
echo "Node:         $SLURM_JOB_NODELIST"

echo "Model:        $MODEL"
echo "Features:     $FEATURE"

echo "Train file:   $TRAIN_FILE"
echo "Test file:    $TEST_FILE"

echo "Model dir:    $MODEL_DIR"
echo "Model file:   $MODEL_FILE"
echo "Output dir:   $OUTPUT_DIR"

echo "======================================================================"


# ----------------------------------------------------------------------
# Check input files
# ----------------------------------------------------------------------

if [ ! -f "$TRAIN_FILE" ]; then
    echo "ERROR: Training file not found:"
    echo "  $TRAIN_FILE"
    exit 1
fi

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
    "$TRAIN_FILE" \
    --target "$MODEL" \
    --features "$FEATURE" \
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


echo ""
echo "Posterior successfully created:"
echo "  $MODEL_FILE"


# ----------------------------------------------------------------------
# 2. Run held-out test
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
echo "Train file:   $TRAIN_FILE"
echo "Test file:    $TEST_FILE"
echo "Posterior:    $MODEL_FILE"
echo "Validation:   $OUTPUT_DIR"

echo "======================================================================"