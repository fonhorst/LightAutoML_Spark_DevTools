#!/usr/bin/env bash

set -ex

# PYSPARK_PYTHON_PATH="/python_envs/.replay_venv/bin/python"
# DRIVER_CORES="2"
# DRIVER_MEMORY="20g"
# DRIVER_MAX_RESULT_SIZE="5g"
# EXECUTOR_INSTANCES="8"
# EXECUTOR_CORES="6"
# EXECUTOR_MEMORY="40g"
# CORES_MAX=$(($EXECUTOR_CORES * $EXECUTOR_INSTANCES))
# N_SAMPLES="5322368"
# # SCRIPT="/submit_files/replay/run_experiment.py"
# SCRIPT="/submit_files/replay/run_experiment_als.py"


spark-submit \
--master yarn \
--deploy-mode cluster \
--conf 'spark.yarn.appMasterEnv.SCRIPT_ENV=cluster' \
--conf 'spark.yarn.appMasterEnv.PYSPARK_PYTHON='${PYSPARK_PYTHON_PATH} \
--conf 'spark.yarn.appMasterEnv.MLFLOW_TRACKING_URI=http://node2.bdcl:8811' \
--conf 'spark.yarn.appMasterEnv.DATASET='${DATASET} \
--conf 'spark.yarn.appMasterEnv.SEED='${SEED} \
--conf 'spark.yarn.appMasterEnv.MODEL='${MODEL} \
--conf 'spark.yarn.appMasterEnv.LOG_TO_MLFLOW='${LOG_TO_MLFLOW} \
--conf "spark.yarn.tags=replay" \
--conf 'spark.kryoserializer.buffer.max=512m' \
--conf 'spark.driver.cores='${DRIVER_CORES} \
--conf 'spark.driver.memory='${DRIVER_MEMORY} \
--conf 'spark.driver.maxResultSize='${DRIVER_MAX_RESULT_SIZE} \
--conf 'spark.executor.instances='${EXECUTOR_INSTANCES} \
--conf 'spark.executor.cores='${EXECUTOR_CORES} \
--conf 'spark.executor.memory='${EXECUTOR_MEMORY} \
--conf 'spark.cores.max='${CORES_MAX} \
--conf 'spark.memory.fraction=0.8' \
--conf 'spark.sql.shuffle.partitions='${CORES_MAX} \
--conf 'spark.default.parallelism='${CORES_MAX} \
--conf 'spark.yarn.maxAppAttempts=1' \
--conf 'spark.sql.autoBroadcastJoinThreshold=100MB' \
--conf 'spark.sql.execution.arrow.pyspark.enabled=true' \
--conf 'spark.scheduler.minRegisteredResourcesRatio=1.0' \
--conf 'spark.scheduler.maxRegisteredResourcesWaitingTime=180s' \
--conf 'spark.eventLog.enabled=true' \
--conf 'spark.eventLog.dir=hdfs://node21.bdcl:9000/shared/spark-logs' \
--conf 'spark.yarn.historyServer.allowTracking=true' \
--conf 'spark.driver.extraJavaOptions=-Dio.netty.tryReflectionSetAccessible=true' \
--conf 'spark.executor.extraJavaOptions=-Dio.netty.tryReflectionSetAccessible=true' \
--py-files '/submit_files/replay/replay_rec-0.9.0-py3-none-any.whl' \
$SCRIPT

# --conf 'spark.executor.extraClassPath='${EXTRA_CLASS_PATH} \
# --conf 'spark.driver.extraClassPath='${EXTRA_CLASS_PATH} \

# --conf 'spark.yarn.appMasterEnv.MLFLOW_TRACKING_URI=http://node2.bdcl:8811' \
# --conf 'spark.yarn.appMasterEnv.USE_SINGLE_DATASET_MODE='${USE_SINGLE_DATASET_MODE} \
# --conf 'spark.yarn.appMasterEnv.CV='${CV} \
# --conf 'spark.yarn.appMasterEnv.DATASET_NAME='${DATASET_NAME} \

# --conf "spark.yarn.tags=USE_SINGLE_DATASET_MODE=${USE_SINGLE_DATASET_MODE},CV=${CV},DATASET_NAME=${DATASET_NAME},EX_INS=${EXECUTOR_INSTANCES},SP_V=${SP_V}" \

# --files '/submit_files/tabular_config.yml' \
# --jars ${JARS} \