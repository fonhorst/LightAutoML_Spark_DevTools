import logging.config
import os
import uuid

import mlflow
from sparklightautoml.computations.utils import get_executors, get_executors_cores
from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from sparklightautoml.pipelines.features.lgb_pipeline import SparkLGBAdvancedPipeline, SparkLGBSimpleFeatures
from sparklightautoml.pipelines.features.linear_pipeline import SparkLinearFeatures
from sparklightautoml.reader.base import SparkToSparkReader
from sparklightautoml.tasks.base import SparkTask
from sparklightautoml.utils import logging_config, VERBOSE_LOGGING_FORMAT, log_exec_timer

from examples_utils import get_spark_session, get_dataset

uid = uuid.uuid4()
logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename=f'/tmp/slama-{uid}.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


feature_pipelines = {
    "linear": SparkLinearFeatures(),
    "lgb_simple": SparkLGBSimpleFeatures(),
    "lgb_adv": SparkLGBAdvancedPipeline()
}


if __name__ == "__main__":
    spark = get_spark_session()

    # settings and data
    # params
    cv = 5
    feat_pipe = os.environ.get("EXP_FEAT_PIPE", "lgb_adv")
    dataset_name = os.environ.get("DATASET", "lama_test_dataset")
    dataset = get_dataset(dataset_name)
    df = dataset.load()

    with mlflow.start_run(experiment_id=os.environ["EXPERIMENT"]):
        # params: count rows + columns
        mlflow.log_params({
            "app_id": spark.sparkContext.applicationId,
            "app_name": spark.sparkContext.appName,
            "dataset": dataset_name,
            "feat_pipe": feat_pipe,
            "exec_instances": len(get_executors()),
            "exec_cores": get_executors_cores()
        })
        mlflow.log_dict(dict(spark.sparkContext.getConf().getAll()), "spark_conf.json")

        task = SparkTask(name=dataset.task_type)
        reader = SparkToSparkReader(task=task, cv=cv, advanced_roles=False)
        feature_pipe = feature_pipelines.get(feat_pipe, None)

        # 1
        with log_exec_timer("reader_time") as reader_timer:
            ds = reader.fit_read(train_data=df, roles=dataset.roles, persistence_manager=PlainCachePersistenceManager())
            persisted_ds = ds.persist()

        mlflow.log_metric(reader_timer.name, reader_timer.duration)

        # 2
        with log_exec_timer("feat_pipe_time") as feat_pipe_timer:
            feat_proc_ds = feature_pipe.fit_transform(persisted_ds)
            feat_proc_ds.data.write.mode('overwrite').format('noop').save()

        mlflow.log_metric(feat_pipe_timer.name, feat_pipe_timer.duration)
