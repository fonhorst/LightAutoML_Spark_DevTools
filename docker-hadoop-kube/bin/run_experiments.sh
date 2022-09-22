#!/usr/bin/env bash

set -ex

declare -A extra_class_path=( ["3.0.1"]="/usr/local/lib/spark_3.0.1_synapseml_0.9.5_jars/*" ["3.1.1"]="/usr/local/lib/spark_3.1.1_synapseml_0.9.5_jars/*" ["3.2.0"]="/usr/local/lib/synapseml_0.9.5_jars/*")
# declare -A extra_class_path=( ["3.0.1"]="/usr/local/lib/spark_3.0.1_synapseml_0.9.5_jars/*" ["3.1.1"]="/usr/local/lib/spark_3.1.1_synapseml_0.9.4_jars/*" ["3.2.0"]="/usr/local/lib/synapseml_0.9.5_jars/*")
declare -A pyspark_python=( ["3.0.1"]="/usr/local/bin/python3.8" ["3.1.1"]="/.venv/bin/python3.9" ["3.2.0"]="/usr/local/bin/python3.9")
# declare -A pyspark_python=( ["3.0.1"]="/usr/local/bin/python3.8" ["3.1.1"]="/.venv_3.1.1_0.9.4/bin/python3.9" ["3.2.0"]="/usr/local/bin/python3.9")
declare -A use_single_dataset_mode=( ["3.0.1"]="True" ["3.1.1"]="True" ["3.2.0"]="True")
declare -A jars=( ["3.0.1"]="/submit_files/spark-lightautoml_2.12-0.1.jar" ["3.1.1"]="/submit_files/spark-lightautoml_2.12-0.1.jar" ["3.2.0"]="/submit_files/spark-lightautoml_2.12-0.1-spark-3.2.0.jar")
# SCRIPT="spark-ml-pipe-lgb-light.py"
SCRIPT="/submit_files/tabular-preset-automl.py"
CV="2"
DRIVER_CORES="2"
DRIVER_MEMORY="20g"
DRIVER_MAX_RESULT_SIZE="5g"
EXECUTOR_CORES="6"
EXECUTOR_MEMORY="27g"

# 3.0.1 3.1.1 3.2.0
for spark_version in 3.1.1 3.2.0
do
    for executor_instances in 4
    do
        for dataset in used_cars_dataset_1x
        do
            CORES_MAX=$(($EXECUTOR_CORES * $executor_instances))
            kubectl -n spark-lama-exps exec spark-submit-$spark_version -- \
            bash -c "export SP_V=$spark_version CORES_MAX=$CORES_MAX EXECUTOR_MEMORY=$EXECUTOR_MEMORY EXECUTOR_CORES=$EXECUTOR_CORES DRIVER_MEMORY=$DRIVER_MEMORY DRIVER_CORES=$DRIVER_CORES DRIVER_MAX_RESULT_SIZE=$DRIVER_MAX_RESULT_SIZE EXECUTOR_INSTANCES=$executor_instances DATASET_NAME=$dataset CV=$CV JARS=${jars[$spark_version]} SCRIPT=$SCRIPT EXTRA_CLASS_PATH=${extra_class_path[$spark_version]} USE_SINGLE_DATASET_MODE=${use_single_dataset_mode[$spark_version]} PYSPARK_PYTHON_PATH=${pyspark_python[$spark_version]} && /submit_files/submit.sh" \
            || continue
        done
    done
done

# nohup ./bin/run_experiments.sh > bin/experiment_logs/run_experiments_spark_submit_24.log  2>&1 &