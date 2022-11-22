#!/usr/bin/env bash

set -ex

PYSPARK_PYTHON_PATH="/python_envs/.replay_venv/bin/python"
DRIVER_CORES="2"
DRIVER_MEMORY="20g"
DRIVER_MAX_RESULT_SIZE="5g"
EXECUTOR_CORES="6"
EXECUTOR_MEMORY="40g"
# N_SAMPLES="5322368"
SCRIPT="/submit_files/replay/run_experiment.py"
SEED="1234"


# MovieLens__1m MovieLens__10m__1552238 MovieLens__10m__5322368 MovieLens__10m MovieLens__20m__18813873 MovieLens__20m
# MovieLens__100k MovieLens__1m MovieLens__10m MovieLens__20m
# MillionSongDataset
for dataset in MillionSongDataset__10000000 MillionSongDataset__6000000 MillionSongDataset__8000000
do
    # 1 2 4 8
    for executor_instances in 4
    do
        # ALS Explicit_ALS SLIM ItemKNN LightFM Word2VecRec PopRec RandomRec_uniform RandomRec_popular_based AssociationRulesItemRec
        for model in ALS
        do
            CORES_MAX=$(($EXECUTOR_CORES * $executor_instances))
            kubectl -n spark-lama-exps exec spark-submit-3.1.1 -- \
            bash -c "export DATASET=$dataset \
            SEED=$SEED \
            MODEL=$model \
            LOG_TO_MLFLOW=True \
            CORES_MAX=$CORES_MAX \
            EXECUTOR_MEMORY=$EXECUTOR_MEMORY \
            EXECUTOR_CORES=$EXECUTOR_CORES \
            DRIVER_MEMORY=$DRIVER_MEMORY \
            DRIVER_CORES=$DRIVER_CORES \
            DRIVER_MAX_RESULT_SIZE=$DRIVER_MAX_RESULT_SIZE \
            EXECUTOR_INSTANCES=$executor_instances \
            SCRIPT=$SCRIPT PYSPARK_PYTHON_PATH=$PYSPARK_PYTHON_PATH \
            && /submit_files/replay/submit.sh" \
            || continue
        done
    done
done

# nohup ./bin/run_replay_experiments.sh > bin/experiment_logs/run_replay_experiments_$(date +%Y-%m-%d-%H%M%S).log  2>&1 &