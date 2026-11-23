#!/bin/bash
# Shared body of the ablation scripts (run_cricket_beta.sh, run_cricket_capacity.sh). The calling script sets the value lists and sources this file.
#
# Every combination of window and setting is one run of train.py (five folds). The runs are dealt
# out to N_TASKS x PROCS workers; finished runs are skipped, so a script can be submitted again.
#
# The windows are tuning windows only. Settings are chosen from the inner validation crickets and
# from label-free numbers (plot_cricket_ablation.py), never from the test folds.

PROCS=${PROCS:-4}
N_TASKS=${SLURM_ARRAY_TASK_COUNT:-1}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
N_WORKERS=$((N_TASKS * PROCS))

# Tuning windows (scaffold block): none of them is a window that is reported as a result.
# run_cricket_eval_windows.sh sets WINDOWS itself (the evaluation windows, frozen settings).
if [[ -z "${WINDOWS+set}" ]]; then
    WINDOWS=("1 139" "2 200" "2 322" "4 98" "1 110" "3 215" "5 53" "7 49" "9 24" "11 81" "13 6" "14 7")
fi

MAX_EPOCHS=${MAX_EPOCHS:-400}
BATCH_SIZE=${BATCH_SIZE:-32}
SEED=${SEED:-1024}

DATA_DIR="/mnt/shared/scratch/zzhang/hawaii_300"                                # read only
CODE_DIR="/mnt/shared/scratch/weiyeli/crop_vae/repos/BETA_VAE_CRICKET"
RESULT_DIR="${RESULT_DIR:-/mnt/shared/scratch/weiyeli/crop_vae/results_cricket/mirror/betavae/ablation}"

source /mnt/apps/users/weiyeli/conda/bin/activate jointvae
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${CODE_DIR}"

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# All combinations: scaffold block beta capacity_z dropout
COMBOS=()
for WINDOW in "${WINDOWS[@]}"; do
    for BETA in "${BETA_VALUES[@]}"; do
        for CZ in "${CAPACITY_VALUES[@]}"; do
            for DROPOUT in "${DROPOUT_VALUES[@]}"; do
                COMBOS+=("${WINDOW} ${BETA} ${CZ} ${DROPOUT}")
            done
        done
    done
done

echo "Started: $(date); task=${TASK_ID}; workers=${N_WORKERS}; runs=${#COMBOS[@]}"

run_worker() {
    local WORKER_ID=$1

    for ((I=WORKER_ID; I<${#COMBOS[@]}; I+=N_WORKERS)); do
        read -r SCAFFOLD BLOCK_ID BETA CZ DROPOUT <<< "${COMBOS[$I]}"

        local SCAFFOLD_DIR="${DATA_DIR}/scaffold_${SCAFFOLD}"
        local SAVE_DIR="${RESULT_DIR}/scaffold_${SCAFFOLD}/block_${BLOCK_ID}/extended"
        SAVE_DIR="${SAVE_DIR}/beta_${BETA}/cz_${CZ}/dropout_${DROPOUT}"
        SAVE_DIR="${SAVE_DIR}/epochs_${MAX_EPOCHS}/seed_${SEED}"

        if [[ -s "${SAVE_DIR}/downstream_summary.csv" ]]; then
            continue
        fi

        mkdir -p "${SAVE_DIR}"

        python -u "${CODE_DIR}/train.py" \
            --out_dir "${SCAFFOLD_DIR}/out_scaffold${SCAFFOLD}_morph_geography" \
            --ld_block_pkl "${SCAFFOLD_DIR}/scaffold_${SCAFFOLD}_merged_ld_blocks.pkl" \
            --save_dir "${SAVE_DIR}" \
            --block_id "${BLOCK_ID}" \
            --use_ext_window \
            --n_splits 5 \
            --inner_val_fraction 0.125 \
            --beta "${BETA}" \
            --capacity_z "${CZ}" \
            --dropout "${DROPOUT}" \
            --max_epochs "${MAX_EPOCHS}" \
            --batch_size "${BATCH_SIZE}" \
            --seed "${SEED}" \
            --device cuda > "${SAVE_DIR}/train.log" 2>&1 \
            || echo "[Failed] ${COMBOS[$I]}; see ${SAVE_DIR}/train.log"

        echo "Finished: $(date); ${COMBOS[$I]}"
    done
}

for ((P=0; P<PROCS; P++)); do
    run_worker $((TASK_ID * PROCS + P)) &
done

wait

echo "Task ${TASK_ID} completed: $(date)"
