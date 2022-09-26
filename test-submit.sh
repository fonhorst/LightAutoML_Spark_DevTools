#!/usr/bin/env bash

set -ex

#export CORE_CONF_fs_defaultFS="hdfs://node21.bdcl:9000"
#export CORE_CONF_hadoop_http_staticuser_user="root"
#export CORE_CONF_hadoop_proxyuser_hue_hosts="*"
#export CORE_CONF_hadoop_proxyuser_hue_groups="*"
#export CORE_CONF_io_compression_codecs="org.apache.hadoop.io.compress.SnappyCodec"
#export YARN_CONF_yarn_log___aggregation___enable="true"
#export YARN_CONF_yarn_log_server_url="http://yarn-history-server:8188/applicationhistory/logs/"
#export YARN_CONF_yarn_resourcemanager_recovery_enabled="true"
#export YARN_CONF_yarn_resourcemanager_store_class="org.apache.hadoop.yarn.server.resourcemanager.recovery.FileSystemRMStateStore"
#export YARN_CONF_yarn_resourcemanager_scheduler_class="org.apache.hadoop.yarn.server.resourcemanager.scheduler.capacity.CapacityScheduler"
#export YARN_CONF_yarn_scheduler_capacity_root_default_maximum___allocation___mb="30720"
#export YARN_CONF_yarn_scheduler_capacity_root_default_maximum___allocation___vcores="8"
#export YARN_CONF_yarn_scheduler_maximum___allocation___mb="53248"
#export YARN_CONF_yarn_scheduler_maximum___allocation___vcores="10"
#export YARN_CONF_yarn_resourcemanager_fs_state___store_uri="/rmstate"
#export YARN_CONF_yarn_resourcemanager_bind___host="0.0.0.0"
#export YARN_CONF_yarn_resourcemanager_hostname="yarn-resourcemanager"
#export YARN_CONF_yarn_resourcemanager_address="yarn-resourcemanager:32182"
#export YARN_CONF_yarn_resourcemanager_scheduler_address="yarn-resourcemanager:31906"
#export YARN_CONF_yarn_resourcemanager_resource__tracker_address="yarn-resourcemanager:30777"
#export YARN_CONF_yarn_timeline___service_enabled="true"
#export YARN_CONF_yarn_timeline___service_generic___application___history_enabled="true"
#export YAEN_CONF_yarn_timeline___service_bind___host="0.0.0.0"
#export YARN_CONF_yarn_timeline___service_hostname="yarn-history-server"
#export YARN_CONF_mapreduce_map_output_compress="true"
#export YARN_CONF_mapred_map_output_compress_codec="org.apache.hadoop.io.compress.SnappyCodec"
#export YARN_CONF_yarn_nodemanager_resource_memory___mb="53248"
#export YARN_CONF_yarn_nodemanager_resource_cpu___vcores="10"
#export YARN_CONF_yarn_nodemanager_disk___health___checker_max___disk___utilization___per___disk___percentage="98.5"
#export YARN_CONF_yarn_nodemanager_remote___app___log___dir="/hadoop/yarn/timeline/hs-app-logs"

export YARN_CONF_DIR="tmp_yarn_conf_dir"

export PYSPARK_PYTHON_PATH="/usr/local/bin/python3.9"

spark-submit \
--master yarn \
--deploy-mode cluster \
--conf 'spark.yarn.appMasterEnv.SCRIPT_ENV=cluster' \
--conf 'spark.yarn.appMasterEnv.MLFLOW_TRACKING_URI=http://node2.bdcl:8811' \
--conf 'spark.yarn.appMasterEnv.PYSPARK_PYTHON='${PYSPARK_PYTHON_PATH} \
--conf "spark.yarn.tags=test-runs" \
--conf "spark.yarn.maxAppAttempts=1" \
--conf 'spark.driver.extraJavaOptions=-Dio.netty.tryReflectionSetAccessible=true -Xss100m' \
--conf 'spark.executor.extraJavaOptions=-Dio.netty.tryReflectionSetAccessible=true -Xss100m' \
--conf 'spark.kryoserializer.buffer.max=512m' \
--conf 'spark.driver.cores=1' \
--conf 'spark.driver.memory=4g' \
--conf 'spark.executor.instances=4' \
--conf 'spark.executor.cores=20' \
--conf 'spark.executor.memory=80g' \
--conf 'spark.cores.max=80' \
--conf 'spark.memory.fraction=0.8' \
--conf 'spark.scheduler.minRegisteredResourcesRatio=1.0' \
--conf 'spark.scheduler.maxRegisteredResourcesWaitingTime=180s' \
--conf 'spark.eventLog.enabled=true' \
--conf 'spark.eventLog.dir=hdfs://node21.bdcl:9000/shared/spark-logs' \
--conf 'spark.yarn.historyServer.allowTracking=true' \
--conf 'spark.sql.autoBroadcastJoinThreshold=-1' \
--conf 'spark.sql.execution.arrow.pyspark.enabled=true' \
--conf 'spark.sql.shuffle.partitions=80' \
--conf 'spark.scheduler.mode=FAIR' \
--jars 'spark-lightautoml_2.12-0.1.jar' \
tmp_highload_join.py
#tmp_complex_merge_join.py
#tmp_parallel_execution.py


#--py-files 'SparkLightAutoML-0.3.0-py3-none-any.whl,examples_utils.py' \
#--conf 'spark.yarn.appMasterEnv.USE_SINGLE_DATASET_MODE='${USE_SINGLE_DATASET_MODE} \
#--conf 'spark.yarn.appMasterEnv.CV='${CV} \
#--conf 'spark.yarn.appMasterEnv.DATASET_NAME='${DATASET_NAME} \
#--conf 'spark.executor.extraClassPath='${EXTRA_CLASS_PATH} \
#--conf 'spark.driver.extraClassPath='${EXTRA_CLASS_PATH} \
#--files 'tabular_config.yml' \