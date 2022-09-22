#!/usr/bin/env bash

set -ex

spark-submit \
--master yarn \
--deploy-mode cluster \
--conf 'spark.executor.extraClassPath='${EXTRA_CLASS_PATH} \
--conf 'spark.driver.extraClassPath='${EXTRA_CLASS_PATH} \
--conf 'spark.yarn.appMasterEnv.SCRIPT_ENV=cluster' \
--conf 'spark.yarn.appMasterEnv.MLFLOW_TRACKING_URI=http://node2.bdcl:8811' \
--conf 'spark.yarn.appMasterEnv.USE_SINGLE_DATASET_MODE='${USE_SINGLE_DATASET_MODE} \
--conf 'spark.yarn.appMasterEnv.CV='${CV} \
--conf 'spark.yarn.appMasterEnv.DATASET_NAME='${DATASET_NAME} \
--conf 'spark.yarn.appMasterEnv.PYSPARK_PYTHON='${PYSPARK_PYTHON_PATH} \
--conf "spark.yarn.tags=USE_SINGLE_DATASET_MODE=${USE_SINGLE_DATASET_MODE},CV=${CV},DATASET_NAME=${DATASET_NAME},EX_INS=${EXECUTOR_INSTANCES},SP_V=${SP_V}" \
--conf 'spark.kryoserializer.buffer.max=512m' \
--conf 'spark.driver.cores='${DRIVER_CORES} \
--conf 'spark.driver.memory='${DRIVER_MEMORY} \
--conf 'spark.driver.maxResultSize='${DRIVER_MAX_RESULT_SIZE} \
--conf 'spark.executor.instances='${EXECUTOR_INSTANCES} \
--conf 'spark.executor.cores='${EXECUTOR_CORES} \
--conf 'spark.executor.memory='${EXECUTOR_MEMORY} \
--conf 'spark.cores.max='${CORES_MAX} \
--conf 'spark.memory.fraction=0.8' \
--conf 'spark.sql.autoBroadcastJoinThreshold=100MB' \
--conf 'spark.sql.execution.arrow.pyspark.enabled=true' \
--conf 'spark.scheduler.minRegisteredResourcesRatio=1.0' \
--conf 'spark.scheduler.maxRegisteredResourcesWaitingTime=180s' \
--conf 'spark.eventLog.enabled=true' \
--conf 'spark.eventLog.dir=hdfs://node21.bdcl:9000/shared/spark-logs' \
--conf 'spark.yarn.historyServer.allowTracking=true' \
--py-files 'SparkLightAutoML-0.3.0-py3-none-any.whl,examples_utils.py' \
--files 'tabular_config.yml' \
--jars 'spark-lightautoml_2.12-0.1.jar' \
$SCRIPT
