#!/usr/bin/env bash

set -ex

spark-submit \
--master yarn \
--deploy-mode cluster \
--conf 'spark.executor.extraClassPath=/root/.ivy2/jars/*' \
--conf 'spark.driver.extraClassPath=/root/.ivy2/jars/*' \
--conf 'spark.yarn.appMasterEnv.SCRIPT_ENV=cluster' \
--conf 'spark.yarn.appMasterEnv.MLFLOW_TRACKING_URI=http://node2.bdcl:8811' \
--conf 'spark.kryoserializer.buffer.max=512m' \
--conf 'spark.driver.cores=2' \
--conf 'spark.driver.memory=20g' \
--conf 'spark.executor.instances=8' \
--conf 'spark.executor.cores=6' \
--conf 'spark.executor.memory=27g' \
--conf 'spark.cores.max=48' \
--conf 'spark.memory.fraction=0.8' \
--conf 'spark.sql.autoBroadcastJoinThreshold=100MB' \
--conf 'spark.sql.execution.arrow.pyspark.enabled=true' \
--conf 'spark.scheduler.minRegisteredResourcesRatio=1.0' \
--conf 'spark.scheduler.maxRegisteredResourcesWaitingTime=180s' \
--conf 'spark.eventLog.enabled=true' \
--conf 'spark.eventLog.dir=hdfs://node21.bdcl:9000/shared/spark-logs' \
--conf 'spark.yarn.historyServer.allowTracking=true' \
--py-files 'SparkLightAutoML-0.3.0.zip,examples_utils.py' \
--files 'tabular_config.yml' \
--jars 'spark-lightautoml_2.12-0.1.jar' \
tabular-preset-automl.py

